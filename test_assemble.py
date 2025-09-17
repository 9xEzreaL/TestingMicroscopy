import os
import queue
import zarr
import threading
import torch
import numpy as np
import tifffile as tiff
from numcodecs import Blosc

from utils.data_utils import DataNormalization
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser
from utils.base_micro_test import InferenceBase, create_tapered_weight_torch




def slicing_data_processer(x0, in_queues, data_info_list, slicing_data_func, NUM_GPUS=1):
    for i, item in enumerate(data_info_list):
        tgt = i % NUM_GPUS
        crd = item["input_crd_idx"]
        arr_np = slicing_data_func(x0, norm=False, crd_x=crd["crd_x"], crd_y=crd["crd_y"], crd_z=crd["crd_z"])

        in_queues[tgt].put((i, item, arr_np))
    for q in in_queues:
        q.put(None)


def writer_thread(save_queue, dest_zarr_proxy):
    enhanced_size = (848, 1056, 848)  # (X, Z, Y)
    chunk_size = (256, 256, 256)

    sr_zarr = zarr.open("/home/tzui/Dataset/results/1dpmLb10downh2NCE10dsp2Try5/DPM/xy.zarr", mode="w",
                        shape=enhanced_size,
                        dtype="float32",
                        chunks=chunk_size,
                        compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE))

    while True:
        item = save_queue.get()
        if item is None:
            break
        crd_x, crd_z, min_y, max_y, one_row = item
        sr_zarr[crd_x[0]:crd_x[1], crd_z[0]:crd_z[1], min_y:max_y] += one_row
        # save_queue.task_done()


class MultiEnhanceWorker(threading.Thread):
    def __init__(self, rank, num_gpus, model, in_q, inputs, result_queue, slicing_data_func,
                 input_augmentation, args, kwargs, S0, S1, S2):
        super().__init__(daemon=True)
        self.rank = rank
        self.num_gpus = num_gpus
        self.args = args
        self.kwargs = kwargs
        self.S0 = S0
        self.S1 = S1
        self.S2 = S2
        self.model = model
        self.model_processer = ModelProcesser(args, kwargs, self.model, kwargs['upsample_params']['size'])
        self.input_augmentation = input_augmentation
        self.in_queue = in_q

        self.inputs = inputs
        self.slicing_data = slicing_data_func
        self.result_queue = result_queue

    def test_ae_decode(self, x0, input_augmentation=[None]):
        assert self.model is not None, "model is None; call get_model first to update model"

        aug_outs = []
        aug_seg_outs = []

        for aug in input_augmentation:
            input_aug = x0 * 1

            if self.args.gpu:
                with torch.cuda.amp.autocast():
                    out, seg = self.model_processer.get_ae_decode(input_aug, aug)
            else:
                out, seg = self.model_processer.get_ae_decode(input_aug, aug)

            aug_outs.append(out)
            aug_seg_outs.append(seg)

        aug_outs = torch.stack(aug_outs, 0)
        out_mean = torch.mean(aug_outs, 0)
        aug_seg_outs = torch.stack(aug_seg_outs, 0)
        aug_seg_outs = torch.mean(aug_seg_outs, 0)
        return out_mean, aug_seg_outs

    def post_process(self, idx, out):
        nx = self.inputs[idx]["input_crd_idx"]["crd_x"]
        ny = self.inputs[idx]["input_crd_idx"]["crd_y"]
        nz = self.inputs[idx]["input_crd_idx"]["crd_z"]

        weight = create_tapered_weight_torch(self.S0, self.S1, self.S2, nz, nx, ny,
                                       size=self.kwargs['assemble_params']['weight_shape'],
                                       device=out.device)

        out *= weight
        return out

    def run(self):
        C0, C1, C2 = self.kwargs['assemble_params']['C']

        with torch.no_grad():
            for in_queue_idx, idx in enumerate(range(self.rank, len(self.inputs), self.num_gpus)):
                _, crd, input = self.in_queue.get()

                if not torch.is_tensor(input):
                    input = torch.from_numpy(input)

                if self.args.fp16:
                    input = input.half()
                input = input.to(f"cuda:{self.rank}", non_blocking=True)

                out_all, _ = self.test_ae_decode(input, self.input_augmentation)
                cropped = out_all[C0:-C0, :, C1:-C1, C2:-C2].squeeze(1)

                cropped = self.post_process(idx=idx, out=cropped).detach().cpu().numpy()

                self.result_queue.put((idx, cropped))



class MicroTest(InferenceBase):
    def __init__(self):
        self.kwargs = None
        # Init model and upsample
        self.model, self.upsample = None, None
        self.normalization = None

    def update_model(self):
        if self.kwargs['model_type'] == 'GAN':
            model_name = os.path.join(self.kwargs['SOURCE'], 'logs', self.kwargs['prj'],
                                      'checkpoints', f"net_g_model_epoch_{self.kwargs['epoch']}.pth")
            print("Loading GAN model from:", model_name)
            self.model = torch.load(model_name, map_location=torch.device('cpu'))
        elif self.kwargs['model_type'] == 'AE':
            component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
            root = os.path.join(self.kwargs['SOURCE'], 'logs', self.kwargs['prj'])
            args_json = read_json_to_args(os.path.join(root, '0.json'))
            model_module = import_model(root, model_name=args_json.models)
            self.model = model_module.GAN(args_json, train_loader=None, eval_loader=None, checkpoints=None)
            self.model = load_pth(self.model, root=root, epoch=self.kwargs['epoch'], model_names=component_names)

        if self.kwargs['model_type'] in ['AE', 'GAN', 'Upsample']:
            self.upsample = torch.nn.Upsample(size=self.kwargs['upsample_params']['size'], mode='trilinear')

        if self.kwargs['model_type'] in ['AE', 'GAN']:
            for param in self.model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                self.model = self.model.half()

    def test_assemble(self, x0, mode="decode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'], saved=None):
        self.create_destination_folder(x0, saved)
        if mode == "decode":
            self._decode_over_from_latent(x0,
                                          destination=os.path.join(self.kwargs['DESTINATION'],
                                                                   self.kwargs["dataset"]),
                                          input_augmentation=input_augmentation,
                                          NUM_GPUS=self.kwargs['N_GPUS'], saved=saved)
        else:
            print(f"not mode supported type : {mode}")

    def _prepare_model_to_multi_gpus(self, NUM_GPUS):
        models = []
        if self.kwargs['model_type'] in ['AE', 'VQQAE']:
            for g in range(NUM_GPUS):
                component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
                if self.kwargs == 'VQQAE':
                    component_names.append("quantize")
                root = os.path.join(self.kwargs['SOURCE'], 'logs', self.kwargs['prj'])
                args_json = read_json_to_args(os.path.join(root, '0.json'))
                model_module = import_model(root, model_name=args_json.models)
                m = model_module.GAN(args_json, train_loader=None, eval_loader=None, checkpoints=None)
                m = load_pth(m, root=root, epoch=self.kwargs['epoch'],
                             model_names=component_names)
                m.to(f'cuda:{g}')
                for param in m.parameters():
                    param.requires_grad = False
                if self.args.fp16:
                    m = m.half()
                models.append(m)
        else:
            raise AttributeError("not support except type AE")
        return models

    def create_destination_folder(self, x0, saved):
        dest = os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"])
        if saved == "zarr":
            dest = os.path.join(dest, "xy.zarr")
            self._create_zarr_destination(x0, destination=dest)
        elif saved == "tiff":
            dest = os.path.join(dest, "xy_tiff")
            os.makedirs(dest, exist_ok=True)
        elif saved == None:
            pass
        else:
            raise AttributeError(f"No attribute saved type {saved}")

    def _create_zarr_destination(self, x0, destination):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # Z, X, Y
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        resolution_ratio = 8
        lr_z, c, lr_x, lr_y, n_z, n_x, n_y = x0.shape
        sr_z = lr_z * resolution_ratio * n_z - 2 * n_z * C0 - (n_z - 1) * S0
        sr_x = lr_x * resolution_ratio * n_x - 2 * n_x * C1 - (n_x - 1) * S1
        sr_y = lr_y * resolution_ratio * n_y - 2 * n_y * C2 - (n_y - 1) * S2
        enhanced_size = (sr_x, sr_z, sr_y) # (X, Z, Y)
        chunk_size = (256, 256, 256)
        if ".zarr" not in destination:
            destination = os.path.join(destination, "xy.zarr")

        os.makedirs(destination, exist_ok=True)
        self.sr_zarr = zarr.open(destination, mode="w",
                          shape=enhanced_size,
                          dtype=self.save_image_datatype,
                          chunks=chunk_size,
                          compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE))

    def _assign_data_to_list(self, x0):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # Z, X, Y
        S0, S1, S2 = self.kwargs['assemble_params']['S']  # S = kwargs['assemble_params']['S']
        resolution_ratio = 8
        lr_z, c, lr_x, lr_y, z, x, y = x0.shape
        data_info_list = []
        for nx in range(x):
            for nz in range(z):
                for ny in range(y):
                    data_info_list.append({"input_crd_idx": {"crd_x": nx if nx != x-1 else -1,
                                                             "crd_y": ny if ny != y-1 else -1,
                                                             "crd_z": nz if nz != z-1 else -1},
                                           "enh_crd": {"crd_x": [(lr_x * resolution_ratio - 2 * C1 - S1) * nx,
                                                                 (lr_x * resolution_ratio - 2 * C1 - S1) * (nx + 1) + S1],
                                                       "crd_y": [(lr_y * resolution_ratio - 2 * C2 - S2) * ny,
                                                                 (lr_y * resolution_ratio - 2 * C2 - S2) * (ny + 1) + S2],
                                                       "crd_z": [(lr_z * resolution_ratio - 2 * C0 - S0) * nz,
                                                                 (lr_z * resolution_ratio - 2 * C0 - S0) * (nz + 1) + S0]
                                                       }})
        return data_info_list

    def _decode_over_from_latent(self, x0, destination, input_augmentation=[None], NUM_GPUS=1, saved=None):
        # pre-compute data coordinate before and after enhancement
        data_info_list = self._assign_data_to_list(x0)

        # stack weight
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        # stupid if not model_clean ex: GAN can not support multiple gpus
        # ToDo : support GAN for multi gpus
        models = self._prepare_model_to_multi_gpus(NUM_GPUS)

        # input / result queue
        result_q = [queue.Queue(maxsize=4) for _ in range(NUM_GPUS)]
        in_q = [queue.Queue(maxsize=4) for _ in range(NUM_GPUS)]
        # slicing data worker
        prod = threading.Thread(target=slicing_data_processer, args=(x0, in_q, data_info_list, self.slicing_data, NUM_GPUS), daemon=False)
        prod.start()

        # launch enhance workers
        workers = []
        for rank in range(NUM_GPUS):
            w = MultiEnhanceWorker(rank=rank, num_gpus=NUM_GPUS, model=models[rank], in_q=in_q[rank], inputs=data_info_list,
                          result_queue=result_q[rank], slicing_data_func=self.slicing_data,
                          input_augmentation=input_augmentation, args=self.args, kwargs=self.kwargs,
                          S0=S0, S1=S1, S2=S2)

            workers.append(w)
        for w in workers:
            w.start()  # start inference

        # main thread collection
        N = len(data_info_list)
        received = 0
        one_row = []
        min_y_crd = None
        while received < N:
            idx, out_cpu = result_q[received % NUM_GPUS].get()
            meta_data = data_info_list[idx]
            nx, ny, nz = meta_data["input_crd_idx"]["crd_x"], meta_data["input_crd_idx"]["crd_y"], meta_data["input_crd_idx"]["crd_z"]
            crd_x, crd_y, crd_z = meta_data["enh_crd"]["crd_x"], meta_data["enh_crd"]["crd_y"], meta_data["enh_crd"]["crd_z"]

            if ny == 0:
                min_y_crd = crd_y[0]
            if ny == -1:
                max_y_crd = crd_y[1]

            out_cpu = np.transpose(out_cpu, (1, 0, 2)) # (X, Z, Y)

            if len(one_row) > 0:
                one_row[-1][:, :, -S2:] = one_row[-1][:, :, -S2:] + out_cpu[:, :, :S2]
                one_row.append(out_cpu[:, :, S2:])
            else:
                one_row.append(out_cpu)

            if ny == -1:
                one_row = np.concatenate(one_row, axis=2)
                if min_y_crd is None:
                    min_y_crd = crd_y[1]

                if saved == "tiff":
                    save = True if nz == -1 else False
                    create = True if nz == 0 else False
                    init = True if nx == 0 else False
                    self.save_to_tiff(one_row, crd_x, crd_z, min_y_crd, max_y_crd, save=save, create=create, init=init, destination=destination)
                elif saved == "zarr":
                    self.save_to_zarr(one_row, crd_x, crd_z, min_y_crd, max_y_crd)
                elif saved == None:
                    pass
                else:
                    raise ValueError(f"No attribute saved type {saved}")
                one_row = []
                min_y_crd = None

            received += 1

        for w in workers:
            w.join()

    def save_to_zarr(self, one_row, crd_x, crd_z, min_y_crd, max_y_crd):
        self.sr_zarr[crd_x[0]:crd_x[1], crd_z[0]:crd_z[1], min_y_crd:max_y_crd] += one_row

    def save_to_tiff(self, one_row, crd_x, crd_z, min_y_crd, max_y_crd, save=False, create=False, init=False, destination=""):
        destination = os.path.join(destination, "xy_tiff")
        os.makedirs(destination, exist_ok=True)
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # Z, X, Y
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        if init:
            self.last = None
        if create:
            resolution_ratio = 8
            lr_z, c, lr_x, lr_y, n_z, n_x, n_y = x0.shape
            sr_z = lr_z * resolution_ratio * n_z - 2 * n_z * C0 - (n_z - 1) * S0
            sr_y = lr_y * resolution_ratio * n_y - 2 * n_y * C2 - (n_y - 1) * S2
            enhanced_size = (one_row.shape[0], sr_z, sr_y)  # (X, Z, Y)
            self.cur_img = np.zeros(enhanced_size)
        if self.last is not None:
            self.cur_img[:S1, :, :] += self.last
            self.last = None
        self.cur_img[:one_row.shape[0], crd_z[0]:crd_z[1], min_y_crd:max_y_crd]+=one_row
        if save:
            for idx in range(self.cur_img.shape[0]):
                slice = self.normalization.backward_normalization(self.cur_img[idx, ::], norm_method=self.kwargs['norm_method'][c], trd=self.kwargs['trd'][c])
                tiff.imwrite(f"{destination}/{str(crd_x[0]+idx).zfill(5)}.tif", slice)# self.cur_img[idx, ::].astype(np.float16))
            self.last = self.cur_img[-S1:, :, :]

    def reslice_ori(self):
        print("="*20+" reslice ori image "+"="*20)
        x0 = self.register_data(get_ori=True)

        up = torch.nn.Upsample(scale_factor=(self.kwargs['N_resolution'], 1), mode='bilinear', align_corners=True)
        zrange = self.kwargs['assemble_params']['zrange']
        yrange = self.kwargs['assemble_params']['yrange']
        xrange = self.kwargs['assemble_params']['xrange']

        C0, C1, C2 = self.kwargs['assemble_params']['C']

        z_start, z_end = (zrange[0])*self.kwargs['N_resolution']+C0, (zrange[-1]+self.kwargs['patch_range']['dx'][0])*self.kwargs['N_resolution']-C0
        y_start, y_end = yrange[0]+C2, yrange[-1]+self.kwargs['patch_range']['dx'][2]-C2
        x_start, x_end = xrange[0]+C1, xrange[-1]+self.kwargs['patch_range']['dx'][1]-C1

        for c in range(len(x0)):
            print('c', c)
            os.makedirs(os.path.join(self.kwargs['DESTINATION'], self.kwargs['dataset'], 'ori_' + str(c) + '/'), exist_ok=True)
            for x in range(x_start, x_end):
                slice = self.normalization.forward_normalization(x0[c][:, x, :], norm_method=self.kwargs['norm_method'][c], trd=self.kwargs['trd'][c])
                slice = up(slice)
                slice = slice[0, 0, z_start:z_end, y_start:y_end]
                slice = self.normalization.backward_normalization(slice, norm_method=self.kwargs['norm_method'][c], trd=self.kwargs['trd'][c])
                tiff.imwrite(os.path.join(self.kwargs['DESTINATION'], self.kwargs['dataset'], 'ori_' + str(c) + '/',
                            f'slice_{x}.tif'), slice)


if __name__ == "__main__":
    tester = MicroTest()
    tester.init_params()
    tester.normalization = DataNormalization(backward_type=tester.save_image_datatype)
    tester.update_model()

    zrange = tester.kwargs['assemble_params']['zrange']
    yrange = tester.kwargs['assemble_params']['yrange']
    xrange = tester.kwargs['assemble_params']['xrange']

    zrange = range(*[eval(str(x)) for x in zrange])
    xrange = range(*[eval(str(x)) for x in xrange])
    yrange = range(*[eval(str(x)) for x in yrange])

    if tester.args.testcube:
        # enhance method
        x0 = tester.register_data() # (32, 4, 32, 32, 5, 4, 4)
        tester.test_assemble(x0, mode="decode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:],
                             saved=tester.args.assemble_method)
    if tester.args.reslice:
        # get corresponding original image
        tester.reslice_ori()


    # python test_assemble.py --config config_chang --augmentation decode --gpu --option VMAT --reslice
    # python test_assemble.py --config config_mr --augmentation decode --gpu --option VMAT --reslice
    # python test_assemble.py --gpu --config aisr122424aedsp2 --augmentation decode --fp16 --option DPM --reslice
    # get_data這邊只能用hbranch, 以及吃原本的原圖