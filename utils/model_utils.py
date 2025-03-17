import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse
import os, importlib, sys

import torch.nn as nn
import time
import tifffile as tiff

def load_pth(gan, root, epoch, model_names):
    for name in model_names:
        setattr(gan, name, torch.load(root + 'checkpoints/' + name + '_model_epoch_' + str(epoch) + '.pth',
                                      map_location=torch.device('cpu')))
    return gan


def import_model(root, model_name):
    model_path = os.path.join(root, f"{model_name}.py")
    module_name = f"dynamic_model_{model_name}"

    # Create the spec
    spec = importlib.util.spec_from_file_location(module_name, model_path)

    # Create the module
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    spec.loader.exec_module(module)

    return module


def read_json_to_args(json_file):
    with open(json_file, 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    return args


class ModelProcesser:
    def __init__(self, args, kwargs, model, upsample_size = None):
        self.args = args
        self.kwargs = kwargs
        self.model = model
        self.upsample_size = upsample_size
        self.gpu = args.gpu
        # dummy method for fix upsample bug
        # self.model.net_g.decoder[0][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))
        # self.model.net_g.decoder[2][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))
        # self.model.net_g.decoder[4][0][0] = Sequential2DUpsampling(scale_factor=(2, 2, 2))

    def get_model_result(self, x0, input_augmentation):
        if self.kwargs['model_type'] == 'AE':
            XupX, Xup = self.get_ae_out(x0, input_augmentation)
        elif self.kwarg['model_type'] == 'GAN':
            XupX, Xup = self.get_gan_out(x0, input_augmentation)
        return XupX, Xup

    def get_ae_out(self, x0, method):
        if self.args.augmentation == "decode":
            Xup, _, hbranch = self.get_ae_encode(x0)

            out_aug = []
            for mc in range(1): # if doing montel carlo for decoder augmentation
                for i, aug in enumerate(method):  # (Z, C, X, Y)
                    XupX = self.get_ae_decode(hbranch, aug)
                    out_aug.append(XupX)
        else:
            out_aug = []
            for i, aug in enumerate(method):  # (Z, C, X, Y)
                Xup, _, hbranch = self.get_ae_encode(x0, aug)
                XupX = self.get_ae_decode(hbranch, aug)
                out_aug.append(XupX)

        out_aug = torch.stack(out_aug, 0)
        XupX = torch.mean(out_aug, 0).cpu()

        return XupX, Xup

    def get_ae_encode(self, x0, method=None):
        if self.gpu:
            x0 = x0.cuda(non_blocking=True)
            # x0 = x0.to('cuda:0', non_blocking=True)
        if self.args.augmentation == "encode":
            x0 = self._test_time_augementation(x0, method=method)

        hb_all = []
        with torch.inference_mode():
            for z in range(0, x0.shape[0], 4):
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    _, posterior, hbranch = self.model.forward(x0[z:z + 4, :, :, :], sample_posterior=False)
                if self.kwargs['hbranchz']:
                    hb_all.append(posterior.sample())
                else:
                    hb_all.append(hbranch)
        hbranch = torch.cat(hb_all, dim=0)

        Xup = torch.nn.Upsample(size=(self.upsample_size[0]*8, self.upsample_size[1], self.upsample_size[2]), mode='trilinear')(
            x0.permute(1, 2, 3, 0).unsqueeze(0))  # (1, C, X, Y, Z)
        Xup = Xup[0, :, ::].permute(3, 0, 1, 2).detach().to('cpu')  # .numpy()  # (Z, C, X, Y))
        del hb_all
        return Xup, _, hbranch.detach().cpu()

    def get_ae_decode(self, hbranch, method):
        if self.gpu:
            hbranch = hbranch.cuda()
        if self.args.augmentation == "decode":
            hbranch = self._test_time_augementation(hbranch, method=method)
            #print("do augmentation on decode for ", method)
        if self.kwargs['hbranchz']:
            hbranch = self.model.decoder.conv_in(hbranch)
        hbranch = hbranch.permute(1, 2, 3, 0).unsqueeze(0)  # (C, X, Y, Z)
        # print("hbranch d : ", hbranch.shape)

        out = self.model.net_g(hbranch, method='decode')
        Xout = out['out0'].detach()#.to('cpu')  # (1, C, X, Y, Z) # , non_blocking=True, non_blocking=True

        #XupX = Xout[0, 0, ::].permute(2, 0, 1)  # .numpy()
        #XupX = self._test_time_augementation(XupX.unsqueeze(1), method=method)
        #XupX = XupX.squeeze()

        #XupX_seg = Xout[0, 1, ::].permute(2, 0, 1)  # .numpy()
        #XupX_seg = self._test_time_augementation(XupX_seg.unsqueeze(1), method=method)
        #XupX_seg = XupX_seg.squeeze()

        # (1, C, X, Y, Z)
        XupX = Xout[0, :].permute(3, 0, 1, 2)  # (Z, C, X, Y)
        XupX = self._test_time_augementation(XupX, method=method)
        return XupX

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