import os
import yaml
import queue
import zarr
import threading
import traceback
from tqdm import tqdm

import torch
import numpy as np
import tifffile as tiff
from numcodecs import Blosc

from utils.data_utils import DataNormalization
from utils.base_micro_test import recreate_volume_folder, InferenceBase

import warnings

warnings.filterwarnings('ignore', message='TiffPage.*read_bytes.*')

import tifffile

warnings.simplefilter('ignore', tifffile.tifffile.TiffFileError)


def writer_thread_func(write_queue, destination, args):
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break  # 終止信號

        mode = item[0]
        try:
            if mode == "decode":
                # 從 test_model 來的結果
                _, iz, ix, iy, out_all_mean, out_all_std, out_seg_all = item
                # if args.reverselog:
                #     out_all_mean = reverse_log(out_all_mean)
                if "xy" in args.save:
                    tiff.imwrite(os.path.join(destination, "xy", f"{iz}_{ix}_{iy}.tif"), out_all_mean)
                if "seg" in args.save:
                    tiff.imwrite(os.path.join(destination, "seg", f"{iz}_{ix}_{iy}.tif"), out_seg_all)
                if int(args.mc) > 1:
                    tiff.imwrite(os.path.join(destination, "xyvar", f"{iz}_{ix}_{iy}.tif"), out_all_std)
            elif mode == "full":
                # 從 test_model 來的結果
                _, iz, ix, iy, out_all_mean, patch = item  # , out_all_std, out_seg_all = item

                # if args.reverselog:
                #     out_all_mean = reverse_log(out_all_mean)
                #     patch = reverse_log(patch)

                # out_all_mean = out_all_mean.mean(axis=-1)
                # out_all_mean = np.permute(out_all_mean, (1, ))
                # (Z, C, X, Y)
                if torch.is_tensor(out_all_mean):
                    out_all_mean = out_all_mean.detach().cpu().numpy()
                if torch.is_tensor(patch):
                    patch = patch.detach().cpu().numpy()

                out_all_mean = np.transpose(out_all_mean, (1, 0, 2, 3))
                patch = np.transpose(patch, (1, 0, 2, 3))

                # TEMP: (-1 ~ 1 ) to (0 255) of unit8
                out_all_mean = ((out_all_mean + 1) / 2 * 255).astype(np.uint8)  # (-1, 1) -> (0, 255)
                patch = ((patch + 1) / 2 * 255).astype(np.uint8)

                if "xy" in args.save:
                    tiff.imwrite(os.path.join(destination, "xy", f"{iz}_{ix}_{iy}.tif"), out_all_mean)
                if "ori" in args.save:
                    tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), patch)
                # if "seg" in args.save:
                #    tiff.imwrite(os.path.join(destination, "seg", f"{iz}_{ix}_{iy}.tif"), out_seg_all)
                # if int(args.mc) > 1:
                #    tiff.imwrite(os.path.join(destination, "xyvar", f"{iz}_{ix}_{iy}.tif"), out_all_std)

            # elif mode == "encode":
            #     # 從 test_ae_encode 來的結果
            #     _, iz, ix, iy, reconstructions, ori, hbranch = item
            #
            #     # if args.reverselog:
            #     #     reconstructions = reverse_log(reconstructions)
            #     #     ori = reverse_log(ori)
            #     if "recon" in args.save:
            #         tiff.imwrite(os.path.join(destination, "recon", f"{iz}_{ix}_{iy}.tif"), reconstructions)
            #     if "ori" in args.save:
            #         tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), ori)
            #     np.save(os.path.join(destination, "hbranch", f"{iz}_{ix}_{iy}.npy"), hbranch)
            else:
                print(f"未知的模式: {mode}")

        except Exception as e:
            print(f"Error writing files (mode={mode}): {e}")
        finally:
            write_queue.task_done()


class MicroTest(InferenceBase):
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None
        self.save_image_datatype = self.args.image_datatype  # uint8 # float32 # uint16
        self.normalization = DataNormalization(backward_type=self.save_image_datatype)

    def test_model(self, x0, input_augmentation=[None], mode="full"):
        assert self.model is not None, "model is None call get_model first to update model"
        result = {}
        # scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.args.gpu)

        d0 = self.kwargs['patch_range']['d0']
        dx = self.kwargs['patch_range']['dx']
        patch = self.slicing_data(x0, crd_x=[d0[1], d0[1] + dx[1]], crd_y=[d0[2], d0[2] + dx[2]], crd_z=[d0[0], d0[0] + dx[0]])
        patch = torch.cat([self._do_upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

        if self.args.fp16 and self.args.gpu:
            patch = patch.half()

        with torch.cuda.amp.autocast():
            if mode == "full":
                # This is for encode+decode
                out, Xup = self.model_processer.get_model_result(patch, input_augmentation)
                result["output"] = out
                result["Xup"] = Xup
            if mode == "encode":
                reconstructions, Xup, hbranch = self.model_processer.get_ae_encode(patch)
                result["reconstructions"] = reconstructions
                result["Xup"] = Xup
                result["hbranch"] = hbranch

        return result

    def test_assemble(self, x0, mode="full", input_augmentation=[None, 'transpose', 'flipX', 'flipY']):
        dz, dx, dy = self.kwargs['assemble_params']['dx_shape']
        zrange = tester.kwargs['assemble_params']['zrange']
        yrange = tester.kwargs['assemble_params']['yrange']
        xrange = tester.kwargs['assemble_params']['xrange']
        zrange = range(*[eval(str(x)) for x in zrange])
        xrange = range(*[eval(str(x)) for x in xrange])
        yrange = range(*[eval(str(x)) for x in yrange])

        if mode == "full":
            recreate_volume_folder(
                destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                folders=["xy", "ori"])
            self._test_over_volume(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                    destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                                    input_augmentation=input_augmentation)
        elif mode == "encode":
            recreate_volume_folder(
                destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                folders=["hbranch"])
            self._test_over_ae_enc_volume(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                               destination=os.path.join(self.kwargs['DESTINATION'],
                                                                        self.kwargs["dataset"]))

    def _test_over_ae_enc_volume(self, x0, dx, dy, dz, zrange, xrange, yrange, destination):
        # x0 : list[npy, proxy]
        N_x = len(xrange)
        N_z = len(zrange)
        N_y = len(yrange)
        # Input (32, 256, 256)
        # This is final size: (32, 4, 32, 32, N_z, N_x, N_y)
        zarr_path = os.path.join(destination, "hbranch.zarr")
        chunk_size = (int(dz/self.kwargs['N_resolution']), 4, int(dx/8), int(dy/8), 1, 1, 1)
        z = zarr.open(zarr_path, mode="w",
                      shape=(int(dz*self.kwargs['N_resolution']/8), 4, int(dx/8), int(dy/8), N_z, N_x, N_y),
                      dtype=np.float64,
                      chunks=chunk_size,
                      compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE))

        for idx, ix in enumerate(xrange):
            for idz, iz in tqdm(enumerate(zrange)):
                for idy, iy in enumerate(yrange):
                    self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                    self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                    result = self.test_model(x0, mode="encode")
                    z[:, :, :, :, idz, idx, idy] = result["hbranch"]

    def _test_over_volume(self, x0, dx, dy, dz, zrange, xrange, yrange, destination,
                           input_augmentation=[None]):
        # writing queue
        write_queue = queue.Queue(maxsize=100)
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in tqdm(xrange):
                for iz in zrange:
                    for iy in yrange:
                        # slicing information by updating kwargs
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # model inference
                        result = self.test_model(x0, input_augmentation)
                        out_all = result["output"]
                        patch = result["Xup"]

                        # writing to queue
                        write_queue.put(("full", iz, ix, iy, out_all, patch))

        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            write_queue.put(None)
            writer_thread.join()

        write_queue.join()

if __name__ == "__main__":
    tester = MicroTest()
    # Update model and upsample
    tester.update_model()
    # Here you can register data
    # register_data get_ori -> Original Image
    # Support tif[2D/3D/latent npy] zarr[3D/latent zarr]
    # x0: 2D/3D -> list[npy, proxy], zarr -> [npy, proxy]
    x0 = tester.register_data(get_ori=True)


    # Test by regions
    # mode : encode or full
    os.makedirs(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"]), exist_ok=True)
    if tester.args.testcube:
        with open(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], 'config.yaml'), 'w') as f:
            yaml.dump(tester.kwargs, f)
        tester.test_assemble(x0, mode="encode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:])





# python test_only_zarr.py --gpu --config aisr122424aedsp_trans --save ori recon xy --augmentation decode --fp16 --option DPM --testcube
# python test_only_zarr.py --gpu --config aisr122424aedsp_trans --augmentation decode --fp16 --option DPM --testcube