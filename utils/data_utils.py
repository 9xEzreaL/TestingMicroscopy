import torch
import numpy as np
from PIL import Image
from skimage import data, io
import matplotlib.pyplot as plt
import json
import argparse
import warnings


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x

# def to_16bit(x, lower_bound=0, upper_bound=550):
#     scaled = (img_array - scale_min) / (scale_max - scale_min)


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    x = x - x.min()
    x = Image.fromarray(to_8bit(x))

    if show:
        io.imshow(np.array(x))
        plt.show()
    if save:
        x.save(save)


def print_num_of_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print('Number of parameters: ' + str(sum([np.prod(p.size()) for p in model_parameters])))


def norm_01(x):
    """
    normalize to 0 - 1
    """
    x = x - x.min()
    x = x / x.max()
    return x

def purge_logs():
    import glob, os
    list_version = sorted(glob.glob('logs/default/*/'))
    list_checkpoint = sorted(glob.glob('logs/default/*/checkpoints/*'))

    checkpoint_epochs = [0] * len(list_version)
    for c in list_checkpoint:
        checkpoint_epochs[list_version.index(c.split('checkpoints')[0])] = int(c.split('epoch=')[-1].split('.')[0])

    for i in range(len(list_version)):
        if checkpoint_epochs[i] < 60:
            os.system('rm -rf ' + list_version[i])


def _CHECK_PARAMS(kwargs, data=None):
    required_path_keys = ['SOURCE', 'root_path', 'DESTINATION']
    for key in required_path_keys:
        if not kwargs.get(key):
            raise ValueError(f"Error: '{key}' is missing or empty in the parameters.")

    if 'assemble_params' not in kwargs:
        warnings.warn("Warning: 'assemble_params' is not provided in the parameters.")
    else:
        assemble_params = kwargs['assemble_params']
        required_assemble_keys = ['C', 'S', 'dx_shape', 'weight_method', 'zrange', 'yrange', 'xrange']
        missing_keys = [key for key in required_assemble_keys if key not in assemble_params]
        if missing_keys:
            raise ValueError(f"Error: The following keys are missing in 'assemble_params': {', '.join(missing_keys)}")
        if assemble_params.get('weight_method') != 'cross':
            raise ValueError("Error: 'weight_method' in 'assemble_params' must be 'cross'.")

        # if data:
        #     # If have data, check size again.
        #     data_z, data_x, data_y = data[0][0, 0, ::].shape
        #     # print(assemble_params["zrange"][1])
        #     # print(kwargs["upsample_params"]["size"][1])
        #     if assemble_params["zrange"][1] + kwargs["upsample_params"]["size"][0] > data_z +1:
        #         assemble_params["zrange"][1] = data_z - kwargs["upsample_params"]["size"][0]  +1
        #         print(f"over range set z axis to {assemble_params['zrange'][1]}")
        #     if assemble_params["xrange"][1] + kwargs["upsample_params"]["size"][1] > data_x +1:
        #         assemble_params["xrange"][1] = data_x - kwargs["upsample_params"]["size"][1]  +1
        #         print(f"over range set x axis to {assemble_params['xrange'][1]}")
        #     if assemble_params["yrange"][1] + kwargs["upsample_params"]["size"][2] > data_y +1:
        #         assemble_params["yrange"][1] = data_y - kwargs["upsample_params"]["size"][2]  +1
        #         print(f"over range set y axis to {assemble_params['yrange'][1]}")
        #
        #
        # # Auto fill params
        # assemble_params["weight_shape"] = [
        #     (x * kwargs['N_resolution'] - 2 * y) if i == 0 else (x - 2 * y)
        #     for i, (x, y) in enumerate(zip(assemble_params['dx_shape'], assemble_params['C']))
        # ]
        # computed_z = assemble_params['dx_shape'][0] - int((assemble_params['C'][0] * 2 + assemble_params['S'][0])/kwargs['N_resolution'])
        # computed_x = assemble_params['dx_shape'][1] - assemble_params['C'][1] * 2 - assemble_params['S'][1]
        # computed_y = assemble_params['dx_shape'][2] - assemble_params['C'][2] * 2 - assemble_params['S'][2]
        #
        # assemble_params['zrange'] = assemble_params['zrange'][:2] + [computed_z]
        # assemble_params['xrange'] = assemble_params['xrange'][:2] + [computed_x]
        # assemble_params['yrange'] = assemble_params['yrange'][:2] + [computed_y]
        
        print(kwargs)


    print("All required parameters are correctly set.")
    return kwargs

"""
{'SOURCE': '/home/ubuntu/Data/TestingMicroscopy', 'root_path': '/home/ubuntu/Data/TestingMicroscopy/DPM4X/', 
'DESTINATION': '/home/tzui/Project/TestingMicroscopy/Dataset/paired_images/', 
'upsample_params': {'size': [32, 256, 256]}, 
'patch_range': {'d0': [189, 120, 400], 'dx': [32, 256, 256]}, 
'assemble_params': {
    'C': [32, 32, 32], 'S': [64, 64, 64], 'dx_shape': [32, 256, 256], 
    'weight_shape': [192, 192, 192], 'weight_method': 'cross', 'zrange': [0, 200, 16], 
    'xrange': [0, 769, 128], 'yrange': [0, 769, 128]
}, 
'norm_method': ['exp', '11'], 
'trd': [[100, 424], [0, 4]], 'dataset': 'DPM4X', 'prj': 'DPM4X/ae/cut/1/', 'epoch': 800, 
'model_type': 'AE', 'hbranchz': True, 
'image_path': ['/ori/3-2ROI000.tif', '/ft0/3-2ROI000.tif'], 
'hbranch_path': '/home/tzui/Project/TestingMicroscopy/Dataset/paired_images/DPM4X/hbranch'}
"""



class DataNormalization:
    def __init__(self, backward_type="float32"):
        self.backward_type = backward_type
        assert self.backward_type in ["float32", "uint16", "uint8"]

    def forward_normalization(self, x0, norm_method, trd):
        if norm_method == 'exp':
            exp_ftr = 7
            x0[x0 <= trd[0]] = trd[0]
            x0[x0 >= trd[1]] = trd[1]
            x0 = np.log10(x0 + 1)
            x0 = np.divide((x0 - x0.mean()), x0.std())
            x0[x0 <= -exp_ftr] = -exp_ftr
            x0[x0 >= exp_ftr] = exp_ftr
            x0 = x0 / exp_ftr
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '11':
            x0[x0 <= trd[0]] = trd[0]
            x0[x0 >= trd[1]] = trd[1]
            x0 = (x0 - x0.min()) / (x0.max() - x0.min())
            x0 = (x0 - 0.5) * 2
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '00':
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        elif norm_method == '01':
            x0 = (x0 - x0.min()) / (x0.max() - x0.min())
            x0 = torch.from_numpy(x0).unsqueeze(0).unsqueeze(0).float()
        return x0

    def backward_normalization(self, x0, norm_method, trd):
        x0 = self._reverse_normalization(x0, norm_method)
        if self.backward_type == "float32":
            return x0
        elif self.backward_type == "uint16":
            if norm_method in ['11', '00', '01']:
                lower_bound, upper_bound = trd[0], trd[1]
            else:
                lower_bound, upper_bound = 0, 550
            return self.to_16bit(x0, lower_bound, upper_bound)
        elif self.backward_type == "uint8":
            return self.to_8bit(x0)

    def _reverse_normalization(self, x0, norm_method):
        if norm_method == '11':
            x0[x0 <= -1] = -1
            x0[x0 >= 1] = 1
            x0 = (x0 + 1) / 2
            return x0
        elif norm_method == '00':
            Warning(f"norm method {norm_method} may potentially caused pixel value issue, if any problem see data_utils.py")
            return x0
        elif norm_method == '01':
            x0[x0 <= 0] = 0
            x0[x0 >= 1] = 1
            return x0

    def to_8bit(self, x0):
        if type(x0) == torch.Tensor:
            x0 = (x0 * 255).numpy().astype(np.uint8)
        else:
            x0 = (x0 * 255).astype(np.uint8)
        return x0

    def to_16bit(self, x0, lower_bound=0, upper_bound=550):
        if type(x0) == torch.Tensor:
            x0 = x0.numpy()
        x0 = x0 * (upper_bound - lower_bound) + lower_bound
        return x0.astype(np.uint16)


if __name__=="__main__":
    params = {'SOURCE': '/home/ubuntu/Data/TestingMicroscopy', 'root_path': '/home/ubuntu/Data/TestingMicroscopy/DPM4X/',
                'DESTINATION': '/home/tzui/Project/TestingMicroscopy/Dataset/paired_images/',
              'N_resolution': 8,
                'upsample_params': {'size': [32, 256, 256]},
                'patch_range': {'d0': [189, 120, 400], 'dx': [32, 256, 256]},
                'assemble_params': {
                    'C': [32, 32, 32], 'S': [64, 64, 64], 'dx_shape': [32, 256, 256],
                    'weight_method': 'cross', 'zrange': [0, 200],
                    'xrange': [0, 769], 'yrange': [0, 769]
                },
                'norm_method': ['exp', '11'],
                'trd': [[100, 424], [0, 4]], 'dataset': 'DPM4X', 'prj': 'DPM4X/ae/cut/1/', 'epoch': 800,
                'model_type': 'AE', 'hbranchz': True,
                'image_path': ['/ori/3-2ROI000.tif', '/ft0/3-2ROI000.tif'],
                'hbranch_path': '/home/tzui/Project/TestingMicroscopy/Dataset/paired_images/DPM4X/hbranch'}
    _CHECK_PARAMS(params)