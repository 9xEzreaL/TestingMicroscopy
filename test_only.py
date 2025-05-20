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
from utils.data_utils import imagesc, DataNormalization, _CHECK_PARAMS
from utils.model_utils import read_json_to_args, import_model, load_pth, ModelProcesser
from utils.base_micro_test import reverse_log, recreate_volume_folder, InferenceBase

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.utils.data import DataLoader, Dataset
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
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16 and self.args.gpu)  # 初始化 scaler

        d0 = self.kwargs['patch_range']['d0']
        dx = self.kwargs['patch_range']['dx']

        patch = [x[:, :, d0[0]:d0[0] + dx[0], d0[1]:d0[1] + dx[1], d0[2]:d0[2] + dx[2]] for x in x0]
        patch = torch.cat([self._do_upsample(x).squeeze().unsqueeze(1) for x in patch], 1)  # (Z, C, X, Y)

        if self.args.fp16 and self.args.gpu:
            patch = patch.half()

        with torch.cuda.amp.autocast():
            if mode == "full":
                out, Xup = self.model_processer.get_model_result(patch, input_augmentation)
                result["output"] = out
                result["Xup"] = Xup
            if mode == "encode":
                reconstructions, Xup, hbranch = self.model_processer.get_ae_encode(patch)
                result["reconstructions"] = reconstructions
                result["Xup"] = Xup
                result["hbranch"] = hbranch

        return result

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
            reconstructions, ori, hbranch = self.model_processer.get_ae_encode(patch)  # (Z, C, X, Y)

        # reshape back to 2d for input
        reconstructions = reconstructions.squeeze().numpy()

        return reconstructions, ori.numpy(), hbranch.detach().to('cpu').numpy() # (Z, X, Y), (Z, X, Y), (X, C, X, Y)

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
            self._test_over_volumne(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                    destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                                    input_augmentation=input_augmentation)
        elif mode == "encode":
            recreate_volume_folder(
                destination=os.path.join(self.kwargs['DESTINATION'], self.kwargs["dataset"]),
                folders=["hbranch"])
            self._test_over_ae_enc_volumne(x0, dx, dy, dz, zrange=zrange, xrange=xrange, yrange=yrange,
                                               destination=os.path.join(self.kwargs['DESTINATION'],
                                                                        self.kwargs["dataset"]))

    def _test_over_ae_enc_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination):
        N_x = len(xrange)
        N_z = len(zrange)
        N_y = len(yrange)

        output = np.empty((32, 4, 32, 32, N_z, N_x, N_y))

        for idx, ix in enumerate(xrange):
            for idz, iz in tqdm(enumerate(zrange)):
                for idy, iy in enumerate(yrange):
                    # 設置 patch_range
                    self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                    self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                    # 模型推理
                    result = self.test_model(x0, mode="encode")

                    output[:, :, :, :, idz, idx, idy] = result["hbranch"]
        # os.makedirs(os.path.join(destination, "hbranch"), exist_ok=True)
        print("save hbranch")
        np.save(os.path.join(destination, "hbranch", f"latent_hbranch.npy"), output)

    def _test_over_volumne(self, x0, dx, dy, dz, zrange, xrange, yrange, destination,
                           input_augmentation=[None]):
        # 初始化寫入隊列和寫入線程
        write_queue = queue.Queue(maxsize=100)  # 控制隊列大小以限制內存使用
        writer_thread = threading.Thread(target=writer_thread_func, args=(write_queue, destination, self.args))
        writer_thread.start()

        try:
            for ix in tqdm(xrange):
                for iz in zrange:
                    for iy in yrange:
                        # 設置 patch_range
                        self.kwargs['patch_range']['d0'] = [iz, ix, iy]
                        self.kwargs['patch_range']['dx'] = [dz, dx, dy]

                        # 模型推理
                        result = self.test_model(x0, input_augmentation)
                        out_all = result["output"]
                        patch = result["Xup"]
                        # SKIP NORMALIZATION
                        if 0:
                            out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=-1),
                                                                                     None,
                                                                                     # self.kwargs["norm_method"][0],
                                                                                     self.kwargs['trd'][0])
                            patch = self.normalization.backward_normalization(patch,
                                                                              None,  # self.kwargs["norm_method"][0],
                                                                              self.kwargs['trd'][0])

                        # 將寫入任務加入隊列
                        write_queue.put(("full", iz, ix, iy, out_all, patch))  # , out_all_std, out_seg_all))
        except Exception as e:
            print(f"Error during processing: {e}")
            traceback.print_exc()
        finally:
            # 所有任務完成後，發送終止信號
            write_queue.put(None)
            writer_thread.join()

        # 確保所有寫入任務完成
        write_queue.join()



if __name__ == "__main__":
    tester = MicroTest()
    # Update model and upsample
    tester.update_model()
    # Here you can register data

    if tester.args.testpatch or tester.args.testcube or tester.args.reslice:
        x0 = tester.get_data(get_ori=True)
        if tester.kwargs.get("norm_mean_std"):
            x0[0] = x0[0] - x0[0].mean()
            x0[0] = x0[0] / x0[0].std()
            x0[0] = x0[0] * tester.kwargs.get("norm_mean_std")[1]
            x0[0] = x0[0] + tester.kwargs.get("norm_mean_std")[0]
        print('Volume shape:  ', print(x0[0].shape))
        print('Volume mean and std', x0[0].mean(), x0[0].std())

    if tester.args.testpatch:
        # 1. Here you can test model with single path image then save it
        result = tester.test_model(x0, [None, 'transpose', 'flipX', 'flipY'][:])
        out = result["output"]
        patch = result["Xup"]

        tester.save_images("out.tif", out, (1, 2, 0, 3), norm_method=None,
                           trd=tester.kwargs['trd'][0])  # norm_method, exp_trd, trd # (Z, C, X, Y, N)
        tester.save_images("patch.tif", patch, (1, 2, 0, 3), norm_method=None,
                           trd=tester.kwargs['trd'][0])  # (Z, C, X, Y)
        # print("Single patch testing time : ", time.time() - tini)

    if tester.args.testcube:
        # recreate_volume_folder(
        #     destination=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"]),
        #     folders=["xy", "ori"])

        # save tester.kwargs to yaml file
        with open(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], 'config.yaml'), 'w') as f:
            yaml.dump(tester.kwargs, f)

        zrange = tester.kwargs['assemble_params']['zrange']
        yrange = tester.kwargs['assemble_params']['yrange']
        xrange = tester.kwargs['assemble_params']['xrange']
        zrange = range(*[eval(str(x)) for x in zrange])
        xrange = range(*[eval(str(x)) for x in xrange])
        yrange = range(*[eval(str(x)) for x in yrange])
        tester.test_assemble(x0, mode="encode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:])


# python test_only.py --config config_chang --save ori xy --augmentation encode --testcube --gpu
# python test_only.py --config config_mr --save ori xy --augmentation decode --testcube --gpu




