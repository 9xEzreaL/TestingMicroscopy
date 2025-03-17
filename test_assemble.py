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
from utils.base_micro_test import reverse_log, recreate_volume_folder, create_tapered_weight, InferenceBase

import zarr
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from torch.utils.data import DataLoader, Dataset


class MicroTest(InferenceBase):
    def __init__(self):
        # Init all args for data and model
        self.init_params()

        # Init model and upsample
        self.model, self.upsample = None, None
        self.save_image_datatype = self.args.image_datatype # uint8 # float32 # uint16
        self.normalization = DataNormalization(backward_type=self.save_image_datatype)


    def test_assemble(self, x0, mode="decode", input_augmentation=[None, 'transpose', 'flipX', 'flipY']):
        if mode == "decode":
            self._test_over_ae_dec_volumne_from_z(x0,
                                               destination=os.path.join(self.kwargs['DESTINATION'],
                                                                        self.kwargs["dataset"], self.args.roi),
                                               input_augmentation=input_augmentation)

    def _test_over_ae_dec_volumne_from_z(self, x0, destination, input_augmentation=[None]):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # C = kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']  # S = kwargs['assemble_params']['S']

        current_x_position = 0
        last_block = None

        z, x, y = x0.shape[4:]

        if not torch.is_tensor(x0):
            x0 = torch.from_numpy(x0)
        else:
            x0 = x0

        if self.args.fp16:
            x0 = x0.half()
        if self.args.gpu:
            x0 = x0.cuda()

        fixed_w = create_tapered_weight(S0, S1, S2, 1, 1, 1, size=self.kwargs['assemble_params']['weight_shape'],
                                        edge_size=64)

        T1 = time.time()
        for nx in tqdm(range(x)):
            one_column = []
            for nz in range(z):
                one_row = []
                for ny in range(y):
                    input = x0[:, :, :, :, nz, nx, ny]
                    # get weight
                    if nx == x - 1:
                        nx = -1
                    if ny == y - 1:
                        ny = -1
                    if nz == z - 1:
                        nz = -1
                    # t1 = time.time()
                    if (nz == 0 or nz == -1) or (nx == 0 or nx == -1) or (ny == 0 or ny == -1):
                        w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                                  size=self.kwargs['assemble_params']['weight_shape'],
                                                  edge_size=64)
                    else:
                        w = fixed_w
                    with torch.cuda.amp.autocast():
                        out_all = self.test_ae_decode(input, input_augmentation)
                    # out_all_mean = self.normalization.backward_normalization(out_all.mean(axis=3),
                    #                                                          self.kwargs["norm_method"][0],
                    #                                                          self.kwargs['exp_trd'][0],
                    #                                                          self.kwargs['trd'][0])
                    # out_all_std = self.normalization.backward_normalization(out_all.std(axis=3),
                    #                                                         self.kwargs["norm_method"][0],
                    #                                                         self.kwargs['exp_trd'][0],
                    #                                                         self.kwargs['trd'][0])

                    cropped = out_all[C0:-C0, :, C1:-C1, C2:-C2].detach().cpu() # Z, C, X, Y
                    w = np.stack([w] * cropped.shape[1], axis=1)
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, :, -S2:] = one_row[-1][:, :, :, -S2:] + cropped[:, :, :, :S2]
                        one_row.append(cropped[:, :, :, S2:])
                    else:
                        one_row.append(cropped)

                one_row = np.concatenate(one_row, axis=3)  # (Z, C, X, Y)
                one_row = np.transpose(one_row, (1, 2, 0, 3))  # (C, X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, :, -S0:, :] = one_column[-1][:, :, -S0:, :] + one_row[:, :, :S0, :]
                    one_column.append(one_row[:, :, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=2).astype(np.float32)  # (C, X, Z, Y)
            if last_block is not None:
                one_column[:, :S1, ::] = one_column[:, :S1, ::] + last_block[:, -S1:, ::]
            if nx == -1: # deal with the last piece of x
                saved_slice_end = one_column.shape[1]
            else:
                saved_slice_end = one_column.shape[1] - S1
            for xx in range(0, saved_slice_end):
                for c in range(one_column.shape[0]):
                    os.makedirs(os.path.join(destination, f"xy_{str(c)}"), exist_ok=True)
                    tiff.imwrite(os.path.join(destination, f"xy_{str(c)}", f'slice_x_{current_x_position + xx}.tif'),
                                 one_column[c, xx, ::].astype(np.dtype(self.save_image_datatype)))

            last_block = one_column
            current_x_position += one_column.shape[1] - S1
        T2 = time.time()
        print("total time cost: ", T2-T1)

    def test_ae_decode(self, hbranch_data, input_augmentation=[None]):
        assert self.model is not None, "model is None; call get_model first to update model"

        if self.args.gpu:
            hbranch_data = hbranch_data.cuda()

        aug_outs = []
        for aug in input_augmentation:
            input_aug = hbranch_data * 1

            if self.args.fp16 and self.args.gpu:
                input_aug = input_aug.half()
                with torch.cuda.amp.autocast():
                    out = self.model_processer.get_ae_decode(input_aug, aug)
            else:
                out = self.model_processer.get_ae_decode(input_aug, aug)

            aug_outs.append(out)

        aug_outs = torch.stack(aug_outs, 0)
        out_mean = torch.mean(aug_outs, 0)
        return out_mean

    def show_or_save_assemble_microscopy(self, zrange, xrange, yrange, source, output_path="tmp.tif", show=True):
        """
        output_path == "" is to show only
        __future__ :
            show images
        """
        if self.args.assemble_method == "tiff":
            # assemble_func = self._assemble_microscopy_volume_memmap
            assemble_func = self.assemble_microscopy_volumne
        elif self.args.assemble_method == "zarr":
            assemble_func = self._assemble_microscopy_volume_zarr_parallel
        else:
            raise KeyError(f"Only support method tiff or zarr, but got {self.args.assemble_method}")
        # Do assemble
        assemble_func(zrange, xrange, yrange, source, output_path)

    def assemble_microscopy_volumne(self, zrange, xrange, yrange, source, output_path):
        C0, C1, C2 = self.kwargs['assemble_params']['C']  # C = kwargs['assemble_params']['C']
        S0, S1, S2 = self.kwargs['assemble_params']['S']  # S = kwargs['assemble_params']['S']

        for c in range(2):
            os.makedirs(output_path + '_' + str(c), exist_ok=True)

        last_block = None
        current_x_position = 0
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
                    cropped = x[:, C0:-C0, C1:-C1, C2:-C2]

                    w = np.stack([w] * cropped.shape[0], axis=0)
                    # ipdb.set_trace()
                    cropped = np.multiply(cropped, w)
                    if len(one_row) > 0:
                        one_row[-1][:, :, :, -S2:] = one_row[-1][:, :, :, -S2:] + cropped[:, :, :, :S2]
                        one_row.append(cropped[:, :, :, S2:])
                    else:
                        one_row.append(cropped)

                #print("one row time : ", time.time() - tini)

                one_row = np.concatenate(one_row, axis=3)  # (C, Z, X, Y)
                one_row = np.transpose(one_row, (0, 2, 1, 3))  # (C, X, Z, Y)

                if len(one_column) > 0:
                    one_column[-1][:, :, -S0:, :] = one_column[-1][:, :, -S0:, :] + one_row[:, :, :S0, :]
                    one_column.append(one_row[:, :, S0:, :])
                else:
                    one_column.append(one_row)

            one_column = np.concatenate(one_column, axis=2).astype(np.float32)  # (C, X, Z, Y)

            if last_block is not None:
                one_column[:, :S1, ::] = one_column[:, :S1, ::] + last_block[:, -S1:, ::]

            for xx in range(0, one_column.shape[1] - S1):
                for c in range(one_column.shape[0]):
                    tiff.imwrite(os.path.join(output_path + '_' + str(c), f'slice_x_{current_x_position + xx}.tif'),
                                 one_column[c, xx, ::].astype(np.dtype(self.save_image_datatype)))

            last_block = one_column
            current_x_position += one_column.shape[1] - S1

    def reslicing_ori(self):
        up = torch.nn.Upsample(scale_factor=(self.kwargs['N_resolution'], 1), mode='bilinear', align_corners=True)
        print("reslicing original image....")
        x0 = tester.get_data(get_ori=True)

        C = self.kwargs['assemble_params']['C']
        z_start, z_end = (zrange[0])*8+C[0], (zrange[-1]+self.kwargs['upsample_params']['size'][0])*8-C[0]
        y_start, y_end = yrange[0]+C[2], yrange[-1]+self.kwargs['upsample_params']['size'][2]-C[2]
        x_start, x_end = xrange[0]+C[1], xrange[-1]+self.kwargs['upsample_params']['size'][1]-C[1]

        for c in range(len(x0)):
            print('c', c)
            os.makedirs(os.path.join(self.kwargs['DESTINATION'], self.kwargs['dataset'], 'ori_' + str(c) + '/'), exist_ok=True)
            x0[c] = (x0[c] - x0[c].min()) / (x0[c].max() - x0[c].min())
            for x in range(x_start, x_end):
                slice = up(x0[c][:, :, :, x, :])
                slice = slice[0, 0, z_start:z_end, y_start:y_end]

                tiff.imwrite(os.path.join(self.kwargs['DESTINATION'], self.kwargs['dataset'], 'ori_' + str(c) + '/',
                            f'slice_{x}.tif'), (slice.numpy() * 255).astype(np.uint8))

if __name__ == "__main__":
    tester = MicroTest()
    tester.update_model()

    zrange = tester.kwargs['assemble_params']['zrange']
    yrange = tester.kwargs['assemble_params']['yrange']
    xrange = tester.kwargs['assemble_params']['xrange']

    zrange = range(*[eval(str(x)) for x in zrange])
    xrange = range(*[eval(str(x)) for x in xrange])
    yrange = range(*[eval(str(x)) for x in yrange])

    if 1:
        x0 = tester.get_data()
        tester.test_assemble(x0, mode="decode", input_augmentation=[None, 'transpose', 'flipX', 'flipY'][:2])

    if tester.args.targets is not None:
        # this can only wit you have small patch
        # targets : VMAT, DPM
        for target in tester.args.targets:
            tester.show_or_save_assemble_microscopy(zrange=zrange, xrange=xrange, yrange=yrange,
                                                    source=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, target + '/'),
                                                    output_path=os.path.join(tester.kwargs['DESTINATION'], tester.kwargs["dataset"], tester.args.roi, target + '_assemble')
                                                    )

    if tester.args.reslice:
        tester.reslicing_ori()
        # up = torch.nn.Upsample(scale_factor=(8, 1), mode='bilinear', align_corners=True)
        # print("reslicing....")
        # x0 = tester.get_data(get_ori=True)
        #
        # C = tester.kwargs['assemble_params']['C']
        # z_start, z_end = (zrange[0])*8+C[0], (zrange[-1]+tester.kwargs['upsample_params']['size'][0])*8-C[0]
        # y_start, y_end = yrange[0]+C[2], yrange[-1]+tester.kwargs['upsample_params']['size'][2]-C[2]
        # x_start, x_end = xrange[0]+C[1], xrange[-1]+tester.kwargs['upsample_params']['size'][1]-C[1]
        #
        # for c in range(len(x0)):
        #     print('c', c)
        #     os.makedirs(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs['dataset'], 'ori_' + str(c) + '/'), exist_ok=True)
        #     x0[c] = (x0[c] - x0[c].min()) / (x0[c].max() - x0[c].min())
        #     for x in range(x_start, x_end):
        #         slice = up(x0[c][:, :, :, x, :])
        #         slice = slice[0, 0, z_start:z_end, y_start:y_end]
        #
        #         tiff.imwrite(os.path.join(tester.kwargs['DESTINATION'], tester.kwargs['dataset'], 'ori_' + str(c) + '/',
        #                     f'slice_{x}.tif'), (slice.numpy() * 255).astype(np.uint8))

    # python test_assemble.py --config config_chang --augmentation decode --gpu --option VMAT --reslice
    # python test_assemble.py --config config_mr --augmentation decode --gpu --option VMAT --reslice
    # get_data這邊只能用hbranch, 以及吃原本的原圖