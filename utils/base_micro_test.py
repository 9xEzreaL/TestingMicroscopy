import argparse
import os
import shutil
import yaml
import zarr
import numpy as np
import torch
import tifffile as tiff
import warnings

from utils.data_utils import DataNormalization, _CHECK_PARAMS
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser

def read_image(path):
    """
    read tiff or zarr by folder/ data naming
    """
    if path.endswith(('.tif', '.tiff')):
        return tiff.imread(path)
    elif path.endswith(('.npy')):
        return np.load(path).astype(np.float32)
    elif '.zarr' in path and os.path.isdir(path):
        z = zarr.open(path, mode='r', use_zarr_fill_value_as_mask=True)
        return z
    else:
        raise ValueError(f"不支援的影像格式: {path}")


def reverse_log(x):
    return np.power(10, x)


def recreate_volume_folder(destination, folders=["xy", "ori", "seg", "recon", "hbranch"]):
    """
    刪除並重建指定的資料夾
    """
    for folder in folders:
        folder_path = os.path.join(destination, folder)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)


def create_tapered_weight(S0, S1, S2, nz, nx, ny, size, edge_size=64) -> np.ndarray:
    """
    產生具有線性 taper 邊緣的 3D 權重立方體

    Args:
        S0, S1, S2 (int): taper 區域大小
        nz, nx, ny (int): 在各方向的索引參數
        size (tuple): 權重立方體的形狀，例如 (Z, X, Y)
        edge_size (int): taper 的邊緣寬度

    Returns:
        np.ndarray: 加入 taper 的權重立方體
    """
    weight = np.ones(size)
    taper_S0 = np.linspace(0, 1, S0)
    taper_S1 = np.linspace(0, 1, S1)
    taper_S2 = np.linspace(0, 1, S2)

    # Z 軸 taper
    if nz != 0 and nz != -2:
        weight[:S0, :, :] *= taper_S0.reshape(-1, 1, 1)
    if nz != -1 and nz != -2:
        weight[-S0:, :, :] *= taper_S0[::-1].reshape(-1, 1, 1)
    # X 軸 taper
    if nx != 0 and nx != -2:
        weight[:, :S1, :] *= taper_S1.reshape(1, -1, 1)
    if nx != -1 and nx != -2:
        weight[:, -S1:, :] *= taper_S1[::-1].reshape(1, -1, 1)
    # Y 軸 taper
    if ny != 0 and ny != -2:
        weight[:, :, :S2] *= taper_S2
    if ny != -1 and ny != -2:
        weight[:, :, -S2:] *= taper_S2[::-1]

    return weight

def create_tapered_weight_torch(S0, S1, S2, nz, nx, ny, size, device):
    weight = torch.ones(size, device=device)
    taper_S0 = torch.linspace(0, 1, S0, device=device)
    taper_S1 = torch.linspace(0, 1, S1, device=device)
    taper_S2 = torch.linspace(0, 1, S2, device=device)

    # Z 軸 taper
    if nz != 0 and nz != -2:
        weight[:S0, :, :] *= taper_S0.view(-1, 1, 1)
    if nz != -1 and nz != -2:
        weight[-S0:, :, :] *= taper_S0.flip(0).view(-1, 1, 1)
    # X 軸 taper
    if nx != 0 and nx != -2:
        weight[:, :S1, :] *= taper_S1.view(1, -1, 1)
    if nx != -1 and nx != -2:
        weight[:, -S1:, :] *= taper_S1.flip(0).view(1, -1, 1)
    # Y 軸 taper
    if ny != 0 and ny != -2:
        weight[:, :, :S2] *= taper_S2
    if ny != -1 and ny != -2:
        weight[:, :, -S2:] *= taper_S2.flip(0)

    return weight


class InferenceBase:
    def update_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default="dpmfull", help='which config file')
        parser.add_argument('--option', type=str, default="VMAT", help='which dataset to use')
        parser.add_argument('--mc', type=str, default=1, help='monte carlo inference, mean over N times')
        parser.add_argument('--testpatch', action='store_true', default=False)
        parser.add_argument('--testcube', action='store_true', default=False)
        parser.add_argument('--gpu', action='store_true', default=False)
        parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')
        parser.add_argument('--save', nargs='+', choices=['ori', 'recon', 'xy'], required=False, help="assign image to save: --save ori recon")
        parser.add_argument('--image_datatype', type=str, default="float32", choices=['float32', 'uint16', 'uint8'])
        parser.add_argument('--augmentation', type=str, default="encode")
        parser.add_argument('--reslice', action='store_true', default=False)
        parser.add_argument('--assemble_method', type=str, default='tiff',
                            help='tiff or zarr method while assemble images')
        parser.add_argument('--targets', nargs='+', default=None, required=False, help="assign target to assemble")
        return parser.parse_args()

    def init_params(self):
        self.args = self.update_args()
        self.save_image_datatype = self.args.image_datatype  # uint8 # float32 # uint16
        # yaml path
        config_path = os.path.join('test', self.args.config + '.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.kwargs = self.process_config(config, self.args.option)

    def process_config(self, config, option):
        """
        combine args option and default
        """
        return _CHECK_PARAMS({**config['DEFAULT'], **config[option]})

    def update_model(self):
        if self.kwargs['model_type'] == 'GAN':
            model_name = os.path.join(self.kwargs['SOURCE'], 'logs', self.kwargs['prj'],
                                      'checkpoints', f"net_g_model_epoch_{self.kwargs['epoch']}.pth")
            print("Loading GAN model from:", model_name)
            self.model = torch.load(model_name, map_location=torch.device('cpu'))
        elif self.kwargs['model_type'] in ['AE', 'VQQAE']:
            component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
            if self.kwargs['model_type'] == 'VQQAE':
                component_names.append('quantize')
            root = os.path.join(self.kwargs['SOURCE'], 'logs', self.kwargs['prj'])
            args_json = read_json_to_args(os.path.join(root, '0.json'))
            model_module = import_model(root, model_name=args_json.models)
            self.model = model_module.GAN(args_json, train_loader=None, eval_loader=None, checkpoints=None)
            self.model = load_pth(self.model, root=root, epoch=self.kwargs['epoch'], model_names=component_names)

        if self.kwargs['model_type'] in ['AE', 'GAN', 'Upsample', 'VQQAE']:
            self.upsample = torch.nn.Upsample(size=self.kwargs['upsample_params']['size'], mode='trilinear')
            if self.args.gpu:
                self.model = self.model.cuda()
                self.upsample = self.upsample.cuda()

        if self.kwargs['model_type'] in ['AE', 'GAN', 'VQQAE']:
            for param in self.model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                self.model = self.model.half()

        self.model_processer = ModelProcesser(self.args, self.kwargs,
                                              self.model, self.kwargs['upsample_params']['size'])

    def register_data(self, get_ori=False):
        """

        :param get_ori: ori image array if True, latent array if False
        :return: tiff and 2D zarr -> array / 3D zarr -> proxy
        """
        x0 = []
        if get_ori:
            if self.kwargs.get("image_path"):
                image_paths = [self.kwargs.get("root_path") + x for x in
                              self.kwargs.get("image_path", [])]
                for i, path in enumerate(image_paths):
                    img = read_image(path)
                    x0.append(img)
            elif self.kwargs.get("image_list_path"): # 2D zarr not suggested
                image_list_path = [os.path.join(self.kwargs.get("root_path"), x)
                                   for x in self.kwargs.get("image_list_path")]
                for num, folder in enumerate(image_list_path):
                    ids = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d)) and not d.startswith(".")]
                    img = np.stack([read_image(os.path.join(folder, id)) for id in ids], axis=0)
                    x0.append(img)
            return x0
        else:
            hbranch_path = self.kwargs.get("hbranch_path")
            if hbranch_path:
                if not ".zarr" in hbranch_path:
                    hbranch_path = os.path.join(hbranch_path, "latent_hbranch.npy")
                return read_image(hbranch_path).astype(np.float32)
            else:
                raise ValueError("not valid data path for hbranch_path")


    def slicing_data(self, x0, norm=True, crd_x=None, crd_y=None, crd_z=None):
        if isinstance(x0, list):
            slice_x0 = []
            for idx, img in enumerate(x0):
                img = img[crd_z[0]: crd_z[1], crd_x[0]: crd_x[1], crd_y[0]: crd_y[1]]
                if norm:
                    img = self.normalization.forward_normalization(
                        img, self.kwargs["norm_method"][idx], self.kwargs['trd'][idx])

                slice_x0.append(img)
            return slice_x0
        else:
            img = x0[:, :, :, :, crd_z, crd_x, crd_y]
            # img = x0[:, :, :, :, crd_z:crd_z+1, crd_x:crd_x+1, crd_y:crd_y+1]
            return img


    def save_images(self, outpath, img, axis=None, norm_method=None, exp_trd=None, trd=None):
        # save_image_method
        directory = os.path.dirname(outpath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if norm_method:
            img = self.normalization.backward_normalization(img, norm_method, exp_trd, trd)
        if axis is not None:
            img = np.transpose(img, axis)
        tiff.imwrite(os.path.join(self.kwargs['root_path'], outpath), img)

    def _test_time_augementation(self, x, method):
        axis_mapping_func = {"Z": 0, "X": 2, "Y": 3}
        # x shape: (Z, C, X, Y)
        if method == None:
            return x
        elif method.startswith('flip'):
            x = torch.flip(x, dims=[axis_mapping_func[method[-1]]])
            return x
        elif method == 'transpose':
            x = x.permute(0, 1, 3, 2)
            return x

    def _do_upsample(self, x):
        assert self.upsample is not None, "upsample not initialized，call update_model()"
        return self.upsample(x)


