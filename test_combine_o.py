import argparse
import glob
import json
import os
import shutil
import sys
import time

import threading
import queue
import ipdb
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
import traceback

import networks
import models
from utils.data_utils import imagesc, DataNormalization
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.utils.data import DataLoader, Dataset

"""
1. Inference model
2. Calculate score
3. Save images only
"""

def reverse_log(x):
    return np.power(10, x)

def create_tapered_weight(S0, S1, S2, nz, nx, ny, size, edge_size: int = 64) -> np.ndarray:
    """
    Create a 3D cube with linearly tapered edges in all directions.

    Args:
        size (int): Size of the cube (size x size x size)
        edge_size (int): Size of the tapered edge section

    Returns:
        np.ndarray: 3D array with tapered weights
    """
    # Create base cube filled with ones
    weight = np.ones(size)

    # Create linear taper from 0 to 1
    # taper = np.linspace(0, 1, edge_size)
    taper_S0 = np.linspace(0, 1, S0)
    taper_S1 = np.linspace(0, 1, S1)
    taper_S2 = np.linspace(0, 1, S2)

    # Z
    if nz != 0:
        weight[:S0, :, :] *= taper_S0.reshape(-1, 1, 1)
    if nz != -1:
        weight[-S0:, :, :] *= taper_S0[::-1].reshape(-1, 1, 1)

    # X
    if nx != 0:
        weight[:, :S1, :] *= taper_S1.reshape(1, -1, 1)
    if nx != -1:
        weight[:, -S1:, :] *= taper_S1[::-1].reshape(1, -1, 1)

    # Y
    if ny != 0:
        weight[:, :, :S2] *= taper_S2
    if ny != -1:
        weight[:, :, -S2:] *= taper_S2[::-1]

    return weight

def recreate_volume_folder(destination, mc=1, folder=["xy", "ori", "seg"]):
    # remove and recreate the folder
    if os.path.exists(os.path.join(destination, 'xy')) and "xy" in folder:
        shutil.rmtree(os.path.join(destination, 'xy'))
    if os.path.exists(os.path.join(destination, 'ori')) and "ori" in folder:
        shutil.rmtree(os.path.join(destination, 'ori'))
    if os.path.exists(os.path.join(destination, 'seg')) and "seg" in folder:
        shutil.rmtree(os.path.join(destination, 'seg'))
    if os.path.exists(os.path.join(destination, 'recon')) and "recon" in folder:
        shutil.rmtree(os.path.join(destination, 'recon'))
    if os.path.exists(os.path.join(destination, 'hbranch')) and "hbranch" in folder:
        shutil.rmtree(os.path.join(destination, 'hbranch'))
    if "xy" in folder:
        os.makedirs(os.path.join(destination, 'xy'), exist_ok=True)
    if "ori" in folder:
        os.makedirs(os.path.join(destination, 'ori'), exist_ok=True)
    if "seg" in folder:
        os.makedirs(os.path.join(destination, 'seg'), exist_ok=True)
    if "recon" in folder:
        os.makedirs(os.path.join(destination, 'recon'), exist_ok=True)
    if "hbranch" in folder:
        os.makedirs(os.path.join(destination, 'hbranch'), exist_ok=True)
    if mc > 1:
        if os.path.exists(os.path.join(destination, 'xyvar')):
            shutil.rmtree(os.path.join(destination, 'xyvar'))
        os.makedirs(os.path.join(destination, 'xyvar'))

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
                _, iz, ix, iy, out_all_mean, patch, out_all_std, out_seg_all = item

                # if args.reverselog:
                #     out_all_mean = reverse_log(out_all_mean)
                #     patch = reverse_log(patch)

                if "xy" in args.save:
                    tiff.imwrite(os.path.join(destination, "xy", f"{iz}_{ix}_{iy}.tif"), out_all_mean)
                if "ori" in args.save:
                    tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), patch)
                if "seg" in args.save:
                    tiff.imwrite(os.path.join(destination, "seg", f"{iz}_{ix}_{iy}.tif"), out_seg_all)
                if int(args.mc) > 1:
                    tiff.imwrite(os.path.join(destination, "xyvar", f"{iz}_{ix}_{iy}.tif"), out_all_std)

            elif mode == "encode":
                # 從 test_ae_encode 來的結果
                _, iz, ix, iy, reconstructions, ori, hbranch = item

                # if args.reverselog:
                #     reconstructions = reverse_log(reconstructions)
                #     ori = reverse_log(ori)
                if "recon" in args.save:
                    tiff.imwrite(os.path.join(destination, "recon", f"{iz}_{ix}_{iy}.tif"), reconstructions)
                if "ori" in args.save:
                    tiff.imwrite(os.path.join(destination, "ori", f"{iz}_{ix}_{iy}.tif"), ori)
                np.save(os.path.join(destination, "hbranch", f"{iz}_{ix}_{iy}.npy"), hbranch)
            else:
                print(f"未知的模式: {mode}")

        except Exception as e:
            print(f"Error writing files (mode={mode}): {e}")
        finally:
            write_queue.task_done()

class MicroTest:
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None
        self.save_image_datatype = self.args.image_datatype # uint8 # float32 # uint16
        self.normalization = DataNormalization(backward_type=self.save_image_datatype)


    def init_params(self):
        # Init / Update Model Args
        self.args = self.update_args()
        # Init / Update Data Args
        self.config, self.kwargs = self.update_data_args('test/' + self.args.config + '.yaml')
        print("kwargs : ", self.kwargs)

    def update_args(self):
        parser = argparse.ArgumentParser()
        # projects
        parser.add_argument('--config', type=str, default="config", help='which config file')
        parser.add_argument('--option', type=str, default="Default", help='which dataset to use')
        parser.add_argument('--prj', type=str, default="/ae/cut/1/", help='name of the project')
        parser.add_argument('--epoch', type=str, default='3000', help='epoch #')
        parser.add_argument('--mc', type=str, default=1, help='monte carlo inference, mean over N times')
        parser.add_argument('--model_type', type=str, default='AE', help='GAN or AE or Upsample or None')
        parser.add_argument('--assemble', action='store_true', default=False)
        parser.add_argument('--hbranchz', action='store_true', default=False) # hidden branch for AE middle layer
        parser.add_argument('--gpu', action='store_true', default=False)
        parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')
        parser.add_argument('--reverselog', action='store_true', default=False)
        parser.add_argument('--assemble_method', type=str, default='tiff', help='tiff or zarr method while assemble images')
        parser.add_argument('--save', nargs='+', choices=['ori', 'recon', 'xy', 'seg'], required=False, help="assign image to save: --save ori recon")
        parser.add_argument('--image_datatype', type=str, default="float32")
        parser.add_argument('--augmentation', type=str, default="encode")
        parser.add_argument('--roi', type=str, default='roiA')

        return parser.parse_args()

    def update_data_args(self, config_name):
        with open(config_name, 'r') as f:
            config = yaml.safe_load(f)

        kwargs = config.get(self.args.option, {})
        if not kwargs:
            raise ValueError(f"Option {self.args.option} not found in the configuration.")
        return config, kwargs

    def update_model(self):
        model = None
        upsample = None
        # get GAN model
        if self.args.model_type == 'GAN':
            model_name = self.config['SOURCE'] + '/logs/' + self.kwargs['dataset'] + self.args.prj + '/checkpoints/net_g_model_epoch_' + str(
                self.args.epoch) + '.pth'
            print(model_name)
            model = torch.load(model_name, map_location=torch.device('cpu'))

            # # rergister model method
            # from utils.model_utils import get_gan_out
            # self.get_model_result = get_gan_out

        # get AE model
        if self.args.model_type == 'AE':
            component_names = ['encoder', 'decoder', 'net_g', 'post_quant_conv', 'quant_conv']
            root = self.config['SOURCE'] + '/logs/' + self.kwargs['dataset'] + self.args.prj
            args = read_json_to_args(root + '0.json')  # load config json file

            # dynamically load module
            model_module = import_model(root, model_name=args.models)
            model = model_module.GAN(args, train_loader=None, eval_loader=None, checkpoints=None)
            model = load_pth(model, root=root, epoch=self.args.epoch, model_names=component_names)

            # # rergister model method
            # from utils.model_utils import get_ae_out
            # self.get_model_result = get_ae_out

        if self.args.model_type in ['AE', 'GAN', 'Upsample']:
            upsample = torch.nn.Upsample(size=self.kwargs['upsample_params']['size'], mode='trilinear')
            if self.args.gpu:
                model = model.cuda()
                upsample = upsample.cuda()

        if self.args.model_type in ['AE', 'GAN']:
            for param in model.parameters():
                param.requires_grad = False
            if self.args.fp16:
                model = model.half()
        self.model = model
        self.upsample = upsample
        self.model_processer = ModelProcesser(self.args, self.model, self.kwargs['upsample_params']['size'])

    def get_data(self, norm=True):
        # 我覺得這要改，有兩種情況，一她只要開圖存，不見得真的要paired data,二 他要inference
        image_path = [x for x in self.kwargs.get("image_path", [])]  # if image path is a file
        image_list_path = self.kwargs.get("image_list_path")  # if image path is a directory
        hbranch_path = self.kwargs.get("hbranch_path")

        x0 = []
        if image_path:
            for i in range(len(image_path)):
                img = tiff.imread(image_path[i])
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][i],
                                       self.kwargs['exp_trd'][i], self.kwargs['exp_ftr'][i], self.kwargs['trd'][i])
                x0.append(img)

        # I'm not sure image list path is used for 2D images or need deal with many 3D images cube
        # new method with 2D loading images
        # assert 2D image is (X, Y) output will be [(Z, X, Y), (Z, X, Y)]
        elif image_list_path:
            for num, i in enumerate(image_list_path):
                ids = sorted(os.listdir(image_list_path[num]))
                img = np.stack([tiff.imread(os.path.join(i, id)) for id in ids], 0)
                if norm:
                    img = self.normalization.forward_normalization(img, self.kwargs["norm_method"][num],
                                       self.kwargs['exp_trd'][num], self.kwargs['exp_ftr'][num], self.kwargs['trd'][num])
                x0.append(img)

        elif hbranch_path:
            return np.load(os.path.join(hbranch_path, "latent_hbranch.npy")).astype(np.float32)
            # class HbranchDataset(Dataset):
            #     def __init__(self, folder):
            #         # 搜尋資料夾下所有 npy 檔案，並根據檔名排序（你也可以根據需要修改排序規則）
            #         self.files = sorted(glob.glob(os.path.join(folder, "*.npy")))
            #         if len(self.files) == 0:
            #             raise ValueError(f"No .npy files found in {folder}")
            #
            #     def __len__(self):
            #         return len(self.files)
            #
            #     def __getitem__(self, idx):
            #         # 載入 npy 檔案，可以視需要加入資料轉換（例如轉成 torch.Tensor）
            #         return np.load(self.files[idx], allow_pickle=True), self.files[idx]

            # hbranch_dataset = HbranchDataset(hbranch_path)
            # 這裡預設 batch_size 為 1，你可以根據需求調整 batch_size 與 shuffle 設定
            # x0 = DataLoader(hbranch_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

        else:
            raise ValueError("No valid image path provided.")

        return x0

    def test_model(self, x0, input_augmentation=[None]):
        assert self.model is not None, "model is None call get_model first to update model"

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.args.gpu)  # 初始化 scaler
        out_all = []
        out_seg_all = []
        for m in range(self.args.mc):
            d0 = self.kwargs['patch_range']['d0']
            dx = self.kwargs['patch_range']['dx']

            patch = [x[:, :, d0[0]:d0[0] + dx[0], d0[1]:d0[1] + dx[1], d0[2]:d0[2] + dx[2]] for x in x0]
            patch = torch.cat([self._do_upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

            if self.args.fp16 and self.args.gpu:
                input = patch.half()
                with torch.cuda.amp.autocast():
                    out, Xup, out_seg = self.model_processer.get_model_result(input, input_augmentation)
            else:
                out, Xup, out_seg = self.model_processer.get_model_result(patch, input_augmentation)

            out_all.append(out.numpy())
            if "seg" in self.args.save:
                out_seg_all.append(out_seg.numpy())

        out_all = np.stack(out_all, axis=3)
        if "seg" in self.args.save:
            out_seg_all = np.stack(out_seg_all, axis=3).mean(axis=3)
            return out_all, Xup.numpy(), out_seg_all

        return out_all, Xup.numpy(), ""

    def test_assemble(self, x0, mode="full", input_augmentation=[None, 'transpose', 'flipX', 'flipY']):
        dz, dx, dy = self.kwargs['assemble_params']['dx_shape']
        zrange = range(*self.kwargs['assemble_params']['zrange'])
        xrange = range(*self.kwargs['assemble_params']['xrange'])
        yrange = range(*self.kwargs['assemble_params']['yrange'])

        if mode == "full":
            recreate_volume_folder(
                destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi),
                mc=self.args.mc,
                folder=["xy", "ori", "seg"])
            self._test_over_volumne(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                   destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi),
                                   input_augmentation=input_augmentation)
        elif mode == "encode":
            recreate_volume_folder(
                destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi),
                folder=["recon", "ori", "hbranch"])
            self._test_over_ae_enc_volumne_new(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                       destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi))
        elif mode == "decode":
            recreate_volume_folder(
                destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi),
                folder=["seg", "xy"])
            self._test_over_ae_dec_volumne_new(x0,
                                           destination=os.path.join(self.config['DESTINATION'], self.kwargs["dataset"], self.args.roi),
                                           input_augmentation=input_augmentation)

    def save_images(self, outpath, img, axis=None, norm_method=None, exp_trd=None, trd=None):
        """
        儲存影像的函數，接受完整的輸出路徑。

        參數：
        - outpath (str): 輸出的檔案路徑，可以包含目錄（例如 "abc/def/xxx.tif"）或僅為檔案名稱（例如 "xxx.tif"）。
        - img (np.ndarray): 要儲存的影像數據。
        - axis (tuple, optional): 影像轉置的軸。
        """
        directory = os.path.dirname(outpath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 處理影像
        # if self.args.reverselog:
        #     img = reverse_log(img)

        if norm_method:
            img = self.normalization.backward_normalization(img, norm_method, exp_trd, trd)

        if axis is not None:
            img = np.transpose(img, axis)

        # 儲存影像
        tiff.imwrite(outpath, img)
    
    def test_ae_encode(self, x0):
        assert self.model is not None, "model is None call get_model first to update model"

        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.args.gpu)  # 初始化 scaler

        d0 = self.kwargs['patch_range']['d0']
        dx = self.kwargs['patch_range']['dx']
        patch = [x[:, :, d0[0]:d0[0] + dx[0], d0[1]:d0[1] + dx[1], d0[2]:d0[2] + dx[2]] for x in x0]
        patch = torch.cat([self._do_upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

        if self.args.fp16 and self.args.gpu:
            patch = patch.half()
            with torch.cuda.amp.autocast():
                reconstructions, ori, hbranch = self.model_processer.get_ae_encode(patch)
        else:
            reconstructions, ori, hbranch = self.model_processer.get_ae_encode(patch)  # (Z, C, X, Y)

        # reshape back to 2d for input
        reconstructions = reconstructions.squeeze().numpy()

        return reconstructions, ori.numpy(), hbranch.detach().to('cpu').numpy() # (Z, X, Y), (Z, X, Y), (X, C, X, Y)

    def test_ae_decode(self, hbranch_data, input_augmentation=[None]):
        assert self.model is not None, "model is None; call get_model first to update model"

        if self.args.gpu:
            hbranch_data = hbranch_data.cuda()

        mc_out = []
        mc_seg = []

        for m in range(self.args.mc):
            aug_outs = []
            aug_seg = []
            for aug in input_augmentation:
                input_aug = hbranch_data * 1

                if self.args.fp16 and self.args.gpu:
                    input_aug = input_aug.half()
                    with torch.cuda.amp.autocast():
                        out, out_seg = self.model_processer.get_ae_decode(input_aug, aug)
                else:
                    out, out_seg = self.model_processer.get_ae_decode(input_aug, aug)

                ## out = self._test_time_augementation(out.unsqueeze(1), method=aug)
                ## out = out.squeeze()

                aug_outs.append(out)
                if "seg" in self.args.save:
                    ## out_seg = self._test_time_augementation(out_seg.unsqueeze(1), method=aug)
                    ## out_seg = out_seg.squeeze()
                    aug_seg.append(out_seg)

            aug_outs = torch.stack(aug_outs, 0)
            out_mean = torch.mean(aug_outs, 0)
            mc_out.append(out_mean.cpu().numpy())
            if "seg" in self.args.save:
                aug_seg = torch.stack(aug_seg, 0)
                seg_mean = torch.mean(aug_seg, 0)
                mc_seg.append(seg_mean.cpu().numpy())

        mc_out = np.stack(mc_out, axis=3)

        if "seg" in self.args.save:
            mc_seg = np.stack(mc_seg, axis=3).mean(axis=3)
        else:
            mc_seg = ""
        return mc_out, mc_seg

    def show_or_save_assemble_microscopy(self, zrange, xrange, yrange, source, output_path="tmp.tif", show=True):
        """
        output_path == "" is to show only
        __future__ :
            show images
        """
        if self.args.assemble_method == "tiff":
            # assemble_func = self.__assemble_microscopy_volumne
            # assemble_func = self._assemble_microscopy_volume_memmap
            assemble_func = self.assemble_microscopy_volumne
        elif self.args.assemble_method == "zarr":
            assemble_func = self._assemble_microscopy_volume_zarr_parallel
        else:
            raise KeyError(f"Only support method tiff or zarr, but got {self.args.assemble_method}")
        # Do assemble
        assemble_func(zrange, xrange, yrange, source, output_path)

    def __assemble_microscopy_volumne(self, zrange, xrange, yrange, source, output_path):
        """
        先把最後大小算出來，算完之後開個依樣大的final圖
        """
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        one_stack = []
        for nx in range(len(xrange)):
            one_column = []
            for nz in range(len(zrange)):
                one_row = []
                for ny in range(len(yrange)):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1

                    iz = zrange[nz]
                    ix = xrange[nx]
                    iy = yrange[ny]

                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)

                    # load and crop
                    x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                    if C0 > 0:
                        cropped = x[C0:-C0, :, :]
                    if C1 > 0:
                        cropped = cropped[:, C1:-C1, :]
                    if C2 > 0:
                        cropped = cropped[:, :, C2:-C2]
                    # ipdb.set_trace()
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, -S2:] = one_row[-1][:, :, -S2:] + cropped[:, :, :S2]
                        one_row.append(cropped[:, :, S2:])
                    else:
                        one_row.append(cropped)

                one_row = np.concatenate(one_row, axis=2)  # (Z, X, Y)
                one_row = np.transpose(one_row, (1, 0, 2))  # (X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, -S0:, :] = one_column[-1][:, -S0:, :] + one_row[:, :S0, :]
                    one_column.append(one_row[:, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=1).astype(np.float32)

            # ipdb.set_trace()
            tiff.imwrite('o.tif', one_column)

            if len(one_stack) > 0:
                one_stack[-1][-S1:, :, :] = one_stack[-1][-S1:, :, :] + one_column[:S1, :, :]
                one_stack.append(one_column[S1:, :, :])
            else:
                one_stack.append(one_column)

        one_stack = np.concatenate(one_stack, axis=0).astype(np.float32)

        tiff.imwrite(source[:-1] + '.tif', one_stack)

    def assemble_microscopy_volumne(self, zrange, xrange, yrange, source, output_path):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # C = kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']  # S = kwargs['assemble_params']['S']

        os.makedirs(output_path, exist_ok=True)

        current_x_position = 0
        last_block = None

        def save_block_to_disk(block, start_x, end_x):
            for x_idx in range(start_x, end_x):
                slice_2d = block[x_idx - start_x, :, :]

                tiff.imwrite(os.path.join(output_path, f'slice_x_{current_x_position + x_idx}.tif'),
                             slice_2d.astype(np.dtype(self.save_image_datatype)))


        def process_block(current_block, is_first=False):
            nonlocal current_x_position, last_block

            if is_first:
                save_block_to_disk(current_block[:-S1], 0, current_block.shape[0] - S1)
                current_x_position += current_block.shape[0] - S1
                return current_block[-S1:]
            else:
                # 處理重疊區域
                overlap_region = last_block + current_block[:S1]  # current_block[:S1] = current_block[:S1, :, :]
                save_block_to_disk(overlap_region, 0, S1)
                current_x_position += S1
                # 處理非重疊區域
                non_overlap = current_block[S1:]
                if non_overlap.shape[0] > S1:  # 如果剩餘部分大於S1，儲存除了最後S1的部分
                    save_block_to_disk(non_overlap[:-S1], 0, non_overlap.shape[0] - S1)
                    current_x_position += non_overlap.shape[0] - S1
                    return non_overlap[-S1:]  # 保留最後S1部分
                return non_overlap  # 如果剩餘部分小於等於S1，全部保留

        one_stack = []
        for nx in tqdm(range(len(xrange))):
            one_column = []
            for nz in range(len(zrange)):
                one_row = []
                for ny in range(len(yrange)):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1

                    iz = zrange[nz]
                    ix = xrange[nx]
                    iy = yrange[ny]

                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)

                    # load and crop
                    x = tiff.imread(source + str(iz) + '_' + str(ix) + '_' + str(iy) + '.tif')
                    cropped = x[C0:-C0, C1:-C1, C2:-C2]
                    # ipdb.set_trace()
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, -S2:] = one_row[-1][:, :, -S2:] + cropped[:, :, :S2]
                        one_row.append(cropped[:, :, S2:])
                    else:
                        one_row.append(cropped)

                one_row = np.concatenate(one_row, axis=2)  # (Z, X, Y)
                one_row = np.transpose(one_row, (1, 0, 2))  # (X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, -S0:, :] = one_column[-1][:, -S0:, :] + one_row[:, :S0, :]
                    one_column.append(one_row[:, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=1).astype(np.float32)

            # ipdb.set_trace()
            tiff.imwrite('o.tif', one_column)

            if last_block is None:
                last_block = process_block(one_column, is_first=True)
            else:
                last_block = process_block(one_column)

        if last_block is not None:
            save_block_to_disk(last_block, 0, last_block.shape[0])

    def _assemble_microscopy_volume_memmap(self, zrange, xrange, yrange, source, output_path="tmp_xy.tif"):
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']

        # 計算最終大圖的尺寸
        total_z = len(zrange) * (self.kwargs['assemble_params']['dx_shape'][0] * 8 - 2 * C0 - S0) + S0 # z 8 倍
        total_x = len(xrange) * (self.kwargs['assemble_params']['dx_shape'][1] - 2 * C1 - S1) + S1
        total_y = len(yrange) * (self.kwargs['assemble_params']['dx_shape'][2] - 2 * C2 - S2) + S2

        with tiff.TiffWriter(output_path, bigtiff=True) as tif:
            empty_slice = np.zeros((total_x, total_y), dtype=self.save_image_datatype)
            for _ in range(total_z):
                tif.write(empty_slice, contiguous=True, dtype=self.save_image_datatype)

        volume = tiff.memmap(output_path, mode='r+', shape=(total_x, total_z, total_y), dtype=self.save_image_datatype)

        # 創建一個內存映射的大數組
        posi_x, posi_y, posi_z = 0, 0, 0

        for nx, ix in enumerate(xrange):
            posi_z = 0
            for nz, iz in enumerate(zrange):
                posi_y = 0
                for ny, iy in enumerate(yrange):
                    # get weight
                    if nx == len(xrange) - 1:
                        nx = -1
                    if ny == len(yrange) - 1:
                        ny = -1
                    if nz == len(zrange) - 1:
                        nz = -1
                    # 加載並裁剪小圖塊
                    filename = os.path.join(source, f"{iz}_{ix}_{iy}.tif")
                    x = tiff.imread(filename)

                    stride_x, stride_y, stride_z = x.shape[1] - 2*C0 - S0, x.shape[2] - 2*C1 - S1, x.shape[0] - 2*C2 - S2
                    if C0 > 0:
                        cropped = x[C0:-C0, :, :]
                    if C1 > 0:
                        cropped = cropped[:, C1:-C1, :]
                    if C2 > 0:
                        cropped = cropped[:, :, C2:-C2]

                    # 應用權重
                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                              size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    cropped = np.multiply(cropped, w)
                    cropped = cropped.transpose(1, 0, 2).astype(np.dtype(self.save_image_datatype))

                    volume[
                    posi_x:posi_x + cropped.shape[1],
                    posi_z:posi_z + cropped.shape[0],
                    posi_y:posi_y + cropped.shape[2]] += cropped
                    posi_y += stride_y
                posi_z += stride_z
            posi_x += stride_x

        # 確保數據寫入磁盤
        del volume

    def _assemble_microscopy_volume_zarr_parallel(self, zrange, xrange, yrange, source,
                                                 output_path="assembled_volume.zarr"):
        C0, C1, C2 = self.kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        dx_shape = self.kwargs['assemble_params']['dx_shape']

        # 計算最終大圖的尺寸
        total_z = len(zrange) * (dx_shape[0] * 8 - 2 * C0 - S0) + S0
        total_x = len(xrange) * (dx_shape[1] - 2 * C1 - S1) + S1
        total_y = len(yrange) * (dx_shape[2] - 2 * C2 - S2) + S2

        # 創建 zarr 文件，設置適當的分塊大小
        store = zarr.DirectoryStore(output_path)
        root = zarr.group(store=store, overwrite=True)
        volume = root.create_dataset(
            'volume',
            shape=(total_z, total_x, total_y),
            dtype=self.save_image_datatype,
            chunks=(256, 256, 256),  # 根據具體情況調整分塊大小
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )

        # 預計算所有權重
        weights_cache = self._precompute_weights(zrange, xrange, yrange)

        # 定義處理 Patch 的函數
        def process_patch(iz, ix, iy):
            C0, C1, C2 = self.kwargs['assemble_params']['C']
            S0, S1, S2 = self.kwargs['assemble_params']['S']
            dx_shape = self.kwargs['assemble_params']['dx_shape']

            filename = os.path.join(source, f"{iz}_{ix}_{iy}.tif")
            if not os.path.exists(filename):
                print(f"文件不存在: {filename}")
                return None
            x = tiff.imread(filename)

            # 裁剪邊界
            cropped = x[C0:-C0, C1:-C1, C2:-C2] if C0 > 0 and C1 > 0 and C2 > 0 else x

            # 應用權重
            w = weights_cache.get((iz, ix, iy))
            if w is not None:
                cropped = np.multiply(cropped, w)
            else:
                print(f"權重不存在: iz={iz}, ix={ix}, iy={iy}")

            # 計算位置
            z_idx = zrange.index(iz)
            x_idx = xrange.index(ix)
            y_idx = yrange.index(iy)

            start_z = z_idx * (dx_shape[0] * 8 - 2 * C0 - S0)
            start_x = x_idx * (dx_shape[1] - 2 * C1 - S1)
            start_y = y_idx * (dx_shape[2] - 2 * C2 - S2)

            # cropped = self.normalization.backward_normalization(cropped, self.kwargs["norm_method"][0],
            #                                                             self.kwargs['exp_trd'][0], self.kwargs['trd'][0])

            return (start_z, start_x, start_y, cropped)

        # 使用 ThreadPoolExecutor 進行並行處理
        with ThreadPoolExecutor(max_workers=8) as executor:
            patch_coords = [(iz, ix, iy) for iz in zrange for ix in xrange for iy in yrange]
            for result in tqdm(executor.map(lambda coords: process_patch(*coords), patch_coords),
                               total=len(patch_coords)):
                if result is None:
                    continue
                start_z, start_x, start_y, cropped = result
                end_z = start_z + cropped.shape[0]
                end_x = start_x + cropped.shape[1]
                end_y = start_y + cropped.shape[2]
                # 將處理後的 Patch 添加到大圖中
                volume[start_z:end_z, start_x:end_x, start_y:end_y] += cropped

        print(f"組裝完成，保存至 {output_path}")

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

    def _test_over_ae_enc_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in xrange:
                for iz in tqdm(zrange):
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        reconstructions, ori, hbranch = self.test_ae_encode(x0)

                        reconstructions = self.normalization.backward_normalization(reconstructions,
                                                                            self.kwargs["norm_method"][0],
                                                                            self.kwargs['exp_trd'][0],
                                                                            self.kwargs['trd'][0])
                        ori = self.normalization.backward_normalization(ori,
                                                                            self.kwargs["norm_method"][0],
                                                                            self.kwargs['exp_trd'][0],
                                                                            self.kwargs['trd'][0])

                        # 將寫入任務加入隊列 "ae",
                        write_queue.put(("encode", iz, ix, iy, reconstructions, ori, hbranch))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _test_over_ae_enc_volumne_new(self, x0, dx, dy, dz, zrange, xrange, yrange, destination):
        N_x = len(xrange)
        N_z = len(zrange)
        N_y = len(yrange)

        output = np.empty((32, 4, 32, 32, N_z, N_x, N_y))

        # try:
        for idx, ix in enumerate(xrange):
            for idz, iz in tqdm(enumerate(zrange)):
                for idy, iy in enumerate(yrange):
                    # 設置 patch_range
                    self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                    self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                    # 模型推理
                    reconstructions, ori, hbranch = self.test_ae_encode(x0)

                    output[:, :, :, :, idz, idx, idy] = hbranch
        np.save(os.path.join(destination, "hbranch", f"latent_hbranch.npy"), output)

    def _test_over_ae_dec_volumne(self, x0, destination, input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        hbranch_iter = iter(x0)
        try:
            for _ in tqdm(range((len(hbranch_iter)))):
                batch = next(hbranch_iter)
    
                hbranch_data, filename = batch
                iz, ix, iy = filename[0].replace(".npy", "").split("/")[-1].split("_")
                if not torch.is_tensor(hbranch_data):
                    hbranch_data = torch.from_numpy(hbranch_data[0])
                else:
                    hbranch_data = hbranch_data[0]
    
                out_all, out_seg_all = self.test_ae_decode(hbranch_data, input_augmentation)
                out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=3),
                                                                         self.kwargs["norm_method"][0],
                                                                         self.kwargs['exp_trd'][0],
                                                                         self.kwargs['trd'][0])
                out_all_std = self.normalization.backward_normalization(out_all.std(axis=3),
                                                                        self.kwargs["norm_method"][0],
                                                                        self.kwargs['exp_trd'][0],
                                                                        self.kwargs['trd'][0])
                # out_seg_all = self.normalization.backward_normalization(out_seg_all,
                #                                                         self.kwargs["norm_method"][0],
                #                                                         self.kwargs['exp_trd'][0],
                #                                                         [0, 255])
                # 將寫入任務加入隊列
                write_queue.put(("decode", iz, ix, iy, out_all_mean, out_all_std, out_seg_all))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _test_over_ae_dec_volumne_new(self, x0, destination, input_augmentation=[None]):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # C = kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']  # S = kwargs['assemble_params']['S']

        current_x_position = 0
        last_block = None
        os.makedirs(os.path.join(destination, "xy_assemble"), exist_ok=True)

        def save_block_to_disk(block, start_x, end_x):
            for x_idx in range(start_x, end_x):
                slice_2d = block[x_idx - start_x, :, :]

                tiff.imwrite(os.path.join(destination, f'slice_x_{current_x_position + x_idx}.tif'),
                             slice_2d.astype(np.dtype(self.save_image_datatype)))

        def process_block(current_block, is_first=False):
            nonlocal current_x_position, last_block

            if is_first:
                save_block_to_disk(current_block[:-S1], 0, current_block.shape[0] - S1)
                current_x_position += current_block.shape[0] - S1
                return current_block[-S1:]
            else:
                # 處理重疊區域
                overlap_region = last_block + current_block[:S1]  # current_block[:S1] = current_block[:S1, :, :]
                save_block_to_disk(overlap_region, 0, S1)
                current_x_position += S1
                # 處理非重疊區域
                non_overlap = current_block[S1:]
                if non_overlap.shape[0] > S1:  # 如果剩餘部分大於S1，儲存除了最後S1的部分
                    save_block_to_disk(non_overlap[:-S1], 0, non_overlap.shape[0] - S1)
                    current_x_position += non_overlap.shape[0] - S1
                    return non_overlap[-S1:]  # 保留最後S1部分
                return non_overlap  # 如果剩餘部分小於等於S1，全部保留

        z, x, y = x0.shape[4:]

        if not torch.is_tensor(x0):
            x0 = torch.from_numpy(x0)
        else:
            x0 = x0
        # time.sleep(10)
        if self.args.fp16 and self.args.gpu:
            x0 = x0.half()
        
        x0 = x0.cuda()
        # print(x0.device)
        # time.sleep(10)
        # assert 0
        fixed_w = create_tapered_weight(S0, S1, S2, 1, 1, 1, size=self.kwargs['assemble_params']['weight_shape'],
                                  edge_size=64)
        T1 = time.time()
        for nx in tqdm(range(x)):
            one_column = []
            for nz in range(z):
                one_row = []
                for ny in range(y):
                    input = x0[:16, :, :, :, nz, nx, ny]
                    # get weight
                    if nx == x - 1:
                        nx = -1
                    if ny == y - 1:
                        ny = -1
                    if nz == z - 1:
                        nz = -1
                    t1 = time.time()
                    if (nz == 0 or nz == -1) or (nx == 0 or nx == -1) or (ny == 0 or ny == -1):
                        w = create_tapered_weight(S0, S1, S2, nz, nx, ny, size=self.kwargs['assemble_params']['weight_shape'],
                                                  edge_size=64)
                    else:
                        w = fixed_w
                    t2 = time.time()
                    print("create_tapered_weight : ", t2-t1)
                    with torch.cuda.amp.autocast():
                        out_all, out_seg_all = self.test_ae_decode(input, input_augmentation)
                    t3 = time.time()
                    print("test_ae_decode : ", t3-t2)
                    # out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=3),
                    #                                                          self.kwargs["norm_method"][0],
                    #                                                          self.kwargs['exp_trd'][0],
                    #                                                          self.kwargs['trd'][0])
                    # out_all_std = self.normalization.backward_normalization(out_all.std(axis=3),
                    #                                                         self.kwargs["norm_method"][0],
                    #                                                         self.kwargs['exp_trd'][0],
                    #                                                         self.kwargs['trd'][0])
                    t4 = time.time()
                    print("backward_normalization : ", t4-t3)
                    cropped = out_all.mean(axis=3)[C0:-C0, C1:-C1, C2:-C2]
                    # cropped = out_all_mean[C0:-C0, C1:-C1, C2:-C2]
                    cropped = np.multiply(cropped, w)
                    t5 = time.time()
                    print("multiply : ", t5-t4)
                    if len(one_row) > 0:
                        one_row[-1][:, :, -S2:] = one_row[-1][:, :, -S2:] + cropped[:, :, :S2]
                        one_row.append(cropped[:, :, S2:])
                    else:
                        one_row.append(cropped)

                one_row = np.concatenate(one_row, axis=2)  # (Z, X, Y)
                one_row = np.transpose(one_row, (1, 0, 2))  # (X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, -S0:, :] = one_column[-1][:, -S0:, :] + one_row[:, :S0, :]
                    one_column.append(one_row[:, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=1).astype(np.float32)

            if last_block is None:
                last_block = process_block(one_column, is_first=True)
            else:
                last_block = process_block(one_column)
        T2 = time.time()
        print("all finished : ", T2-T1)
        if last_block is not None:
            save_block_to_disk(last_block, 0, last_block.shape[0])
        T3 = time.time()
        print("save_block_to_disk : ", T3-T2)


    def _test_over_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination,
                           input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in xrange:
                for iz in tqdm(zrange):
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        out_all, patch, out_seg_all = self.test_model(x0, input_augmentation)

                        out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=3),
                                                                                 self.kwargs["norm_method"][0],
                                                                                 self.kwargs['exp_trd'][0],
                                                                                 self.kwargs['trd'][0])
                        patch = self.normalization.backward_normalization(patch,
                                                                          self.kwargs["norm_method"][0],
                                                                          self.kwargs['exp_trd'][0],
                                                                          self.kwargs['trd'][0])
                        out_all_std = self.normalization.backward_normalization(out_all.std(axis=3),
                                                                                self.kwargs["norm_method"][0],
                                                                                self.kwargs['exp_trd'][0],
                                                                                self.kwargs['trd'][0])
                        # out_seg_all = self.normalization.backward_normalization(out_seg_all,
                        #                                                         self.kwargs["norm_method"][0],
                        #                                                         self.kwargs['exp_trd'][0],
                        #                                                         [0, 255])

                        # 將寫入任務加入隊列
                        write_queue.put(("full", iz, ix, iy, out_all_mean, patch, out_all_std, out_seg_all))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()

    def _precompute_weights(self, zrange, xrange, yrange):
        S0, S1, S2 = self.kwargs['assemble_params']['S']
        weights_cache = {}
        for iz in zrange:
            for ix in xrange:
                for iy in yrange:
                    nz = zrange.index(iz)
                    nx = xrange.index(ix)
                    ny = yrange.index(iy)
                    w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                              size=self.kwargs['assemble_params']['weight_shape'],
                                              edge_size=64)
                    weights_cache[(iz, ix, iy)] = w
        return weights_cache

    def _do_upsample(self, x0):
        assert self.upsample is not None, "upsample is None call get_model first to update upsample"
        return self.upsample(x0)


if __name__ == "__main__":
    tester = MicroTest()
    # Update model and upsample
    tester.update_model()
    # Here you can register data
    x0 = tester.get_data()

    # 1. Here you can test model with single path image then save it
    # out, patch, out_seg = tester.test_model(x0, [None, 'transpose', 'flipX', 'flipY'])
    # tester.save_images("out.tif", out.mean(axis=3), (1, 0, 2), norm_method=tester.kwargs["norm_method"][0], exp_trd=tester.kwargs['exp_trd'][0],
    #                     trd=tester.kwargs['trd'][0]) # norm_method, exp_trd, trd
    # tester.save_images("out_seg.tif", out_seg, (1, 0, 2), norm_method="11", trd=[0, 255])
    # tester.save_images("patch.tif", patch, (1, 0, 2), norm_method=tester.kwargs["norm_method"][0], exp_trd=tester.kwargs['exp_trd'][0],
    #                         trd=tester.kwargs['trd'][0])

    # reconstructions, ori, hbranch = tester.test_ae_encode(x0)

    # 2. Do test assemble then save patch
    # test_assemble -> mode : encode, decode, full

    tester.test_assemble(x0, mode="decode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:2]) #

    # 3. show or save assemble big image from pattch
    # zrange = range(*tester.kwargs['assemble_params']['zrange'])
    # xrange = range(*tester.kwargs['assemble_params']['xrange'])
    # yrange = range(*tester.kwargs['assemble_params']['yrange'])
    #
    # tester.show_or_save_assemble_microscopy(zrange=zrange, xrange=xrange, yrange=yrange,
    #                                         source=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'xy/'),
    #                                         # output_path="tmp_xy.tif",
    #                                         output_path=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'xy_assemble')#
    #                                         )
    #
    # tester.show_or_save_assemble_microscopy(zrange=zrange, xrange=xrange, yrange=yrange,
    #                                         source=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'ori/'),
    #                                         # output_path="tmp_ori.tif",
    #                                         output_path=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'ori_assemble')
    #                                         )
    #
    # tester.show_or_save_assemble_microscopy(zrange=zrange, xrange=xrange, yrange=yrange,
    #                                         source=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'seg/'),
    #                                         output_path=os.path.join(tester.config['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, 'seg_assemble')
    #                                         )
    # t5 = time.time()
    # print("img3 : ", t5-t4)



    # python test_combine_o.py  --prj /ae/cut/1/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --assemble --assemble_method tiff
    # python test_combine_o.py --prj /1dpm/ --epoch 1100 --model_type AE --gpu --hbranchz --assemble --assemble_method tiff --config config_122924 --save_seg
    # CUDA_VISIBLE_DEVICES=3 python test_combine_o.py --prj /1vmat/ --epoch 1100 --model_type AE --gpu --hbranchz --assemble --assemble_method tiff --config config_122924_operate_vmat --save_seg
    # CUDA_VISIBLE_DEVICES=3 python test_combine_o.py --prj /1dpm/ --epoch 1100 --model_type AE --gpu --hbranchz --assemble --assemble_method tiff --config config_122924 --save ori seg xy
    # python test_combine_o.py  --prj /ae/cut/1/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --save ori seg xy
    # python test_combine_o.py  --prj /1dpm/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --save ori seg xy --config config_122924

    # python test_combine_o.py  --prj /ae/cut/1/ --epoch 800 --model_type AE --gpu --hbranchz --reverselog --save ori seg xy --augmentation decode