import argparse
import os
import shutil
import yaml
import numpy as np
import torch
import tifffile as tiff
import warnings

from utils.data_utils import DataNormalization, _CHECK_PARAMS
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser


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
        parser.add_argument('--image_datatype', type=str, default="float32")
        parser.add_argument('--augmentation', type=str, default="encode")
        parser.add_argument('--reslice', action='store_true', default=False)
        parser.add_argument('--host', type=str, default='dummy')
        parser.add_argument('--port', type=str, default='dummy')
        parser.add_argument('--assemble_method', type=str, default='tiff',
                            help='tiff or zarr method while assemble images')
        parser.add_argument('--roi', type=str, default='')
        parser.add_argument('--targets', nargs='+', default=None, required=False, help="assign target to assemble")
        return parser.parse_args()

    def init_params(self):
        self.args = self.update_args()
        self.save_image_datatype = self.args.image_datatype  # uint8 # float32 # uint16
        # 假設 YAML 檔放在 test/ 目錄下，檔名為 {config}.yaml
        config_path = os.path.join('test', self.args.config + '.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.kwargs = self.process_config(config, self.args.option)

    def process_config(self, config, option):
        """
        合併設定檔中的 DEFAULT 與指定 option 的設定
        """
        return _CHECK_PARAMS({**config['DEFAULT'], **config[option]})

    def update_model(self):
        """
        根據 kwargs 中的 model_type 載入模型與建立 upsample 模組
        """
        print(self.kwargs)
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
            if self.args.gpu:
                self.model = self.model.cuda()
                self.upsample = self.upsample.cuda()

        if self.kwargs['model_type'] in ['AE', 'GAN']:
            for param in self.model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                self.model = self.model.half()

        self.model_processer = ModelProcesser(self.args, self.kwargs,
                                              self.model, self.kwargs['upsample_params']['size'])

    def get_data(self, norm=True, get_ori=False):
        """
        讀取資料：
         - 若 get_ori 為 True，則讀取原始影像（從 image_path 或 image_list_path）
         - 否則讀取 hbranch 潛在資料（需指定 hbranch_path）
        """
        x0 = []
        if get_ori:
            if self.kwargs.get("image_path"):
                image_paths = [self.kwargs.get("root_path") + x for x in
                              self.kwargs.get("image_path", [])]
                for i, path in enumerate(image_paths):
                    img = tiff.imread(path)
                    if norm:
                        img = self.normalization.forward_normalization(
                            img, self.kwargs["norm_method"][i], self.kwargs['trd'][i])
                    x0.append(img)
            elif self.kwargs.get("image_list_path"):
                image_list_path = [os.path.join(self.kwargs.get("root_path"), x)
                                   for x in self.kwargs.get("image_list_path")]
                for num, folder in enumerate(image_list_path):
                    ids = sorted(os.listdir(folder))
                    img = np.stack([tiff.imread(os.path.join(folder, id)) for id in ids], axis=0)
                    if norm:
                        img = self.normalization.forward_normalization(
                            img, self.kwargs["norm_method"][num], self.kwargs['trd'][num])
                    x0.append(img)
            # update again
            self.kwargs = _CHECK_PARAMS(self.kwargs, x0)
        else:
            hbranch_path = self.kwargs.get("hbranch_path")
            if hbranch_path:
                x0 = np.load(os.path.join(hbranch_path, "latent_hbranch.npy")).astype(np.float32)
            else:
                raise ValueError("未提供有效的資料路徑 (hbranch_path)")
        return x0

    def save_images(self, outpath, img, axis=None, norm_method=None, exp_trd=None, trd=None):
        """
        儲存影像：
          - 可做 normalization 與轉置 (axis)
        """
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
        assert self.upsample is not None, "upsample 尚未初始化，請先呼叫 update_model()"
        return self.upsample(x)


