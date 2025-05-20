import os
import sys
import itertools
import math
import zarr
import argparse
from io import StringIO
import napari
import torch
import shutil
import numpy as np
from scipy.ndimage import zoom
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton, QFileDialog, QMessageBox,
    QHBoxLayout, QPlainTextEdit, QFormLayout, QLineEdit, QCheckBox, QComboBox, QApplication
)
from qtpy.QtCore import Qt, Signal, QThread, QEvent
from qtpy.QtWidgets import QToolTip, QScrollArea

from test_assemble import MicroTest
from utils.base_micro_test import create_tapered_weight
from utils.data_utils import _CHECK_PARAMS, DataNormalization
import tifffile as tiff


class ModelLoaderThread(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, image_processer, parent=None):
        super().__init__(parent)
        self.image_processer = image_processer

    def run(self):
        try:
            model = self.image_processer.update_model()
            self.finished.emit(model)
        except Exception as e:
            self.error.emit(str(e))


class UpsampleCopyWorker(QThread):
    # Signal emitted when each chunk is copied (returns updated region)
    progress = Signal(object)
    finished = Signal()
    error = Signal(str)

    def __init__(self, src_array, dst_array, upsample_factor=8, parent=None):
        super().__init__(parent)
        self.src_array = src_array  # Original lazy_zarr, shape = (x, z, y)
        self.dst_array = dst_array  # Target zarr store, shape = (x, z*upsample_factor, y)
        self.upsample_factor = upsample_factor  # e.g. 8

    def run(self):
        try:
            # Get original shape and chunks
            src_shape = self.src_array.shape  # (x, z, y)
            chunks = self.src_array.chunks  # e.g. (x_chunk, z_chunk, y_chunk)
            # Keep x and y axes unchanged; z axis is upsampled by upsample_factor
            z_chunk, x_chunk, y_chunk = chunks

            counter = 0
            progress_interval = 10
            for i in range(0, src_shape[0], x_chunk):
                for j in range(0, src_shape[2], y_chunk):
                    for k in range(0, src_shape[1], z_chunk):
                        # Define source block
                        src_slice = (slice(i, min(i + x_chunk, src_shape[0])),
                                     slice(k, min(k + z_chunk, src_shape[1])),
                                     slice(j, min(j + y_chunk, src_shape[2])))
                        block = self.src_array[src_slice]  # shape: (x_block, z_block, y_block)
                        # Upsample z axis using np.repeat (simple method, not high-quality interpolation)
                        upsampled_block = np.repeat(block, self.upsample_factor, axis=1)
                        # Destination slice:
                        # x and y axes remain unchanged; z axis: start index multiplied by upsample_factor
                        dst_slice = (slice(i, min(i + x_chunk, src_shape[0])),
                                     slice(k * self.upsample_factor, min((k + z_chunk) * self.upsample_factor,
                                                                         src_shape[1] * self.upsample_factor)),
                                     slice(j, min(j + y_chunk, src_shape[2])))
                        self.dst_array[dst_slice] = upsampled_block
                        counter += 1
                        if counter % progress_interval == 0:
                            # Only emit signal every progress_interval blocks
                            self.progress.emit(dst_slice)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class EnhanceWorker(QThread):
    # Signal emitted when each patch is processed, returns patch_index and processed result
    patch_finished = Signal(int, object, object, int, int, int, int, bool, bool)
    finished = Signal()
    error = Signal(str)

    def __init__(self, enhance_func, latent, patch_indices, augmentation, parent=None,
                 nz_start=0, nz_end=1, ny_start=0, ny_end=1, S0=64):
        """
        enhance_func: Function that processes enhancement operations
        latent: numpy array or torch tensor, original latent data
        patch_indices: list of patch indices to process (e.g. x-axis patch indices)
        augmentation: parameter passed to iter_enhance_by_yz_slice
        nz_start, nz_end, ny_start, ny_end: Region boundaries
        """
        super().__init__(parent)
        self.enhance_func = enhance_func
        self.latent = latent
        self.patch_indices = patch_indices
        self.augmentation = augmentation
        self.nz_start = nz_start
        self.nz_end = nz_end
        self.ny_start = ny_start
        self.ny_end = ny_end
        self.S0 = S0

    def run(self):
        try:
            last_patch_x = None
            import time
            for nx, patch_idx in enumerate(self.patch_indices):
                # Determine edge conditions
                if len(self.patch_indices) == 1:
                    is_x_up_edge = True
                    is_x_down_edge = True
                    nx = -2
                elif nx == 0:
                    is_x_up_edge = True
                    is_x_down_edge = False
                elif nx == len(self.patch_indices) - 1:
                    nx = -1
                    is_x_up_edge = False
                    is_x_down_edge = True
                else:
                    is_x_up_edge = False
                    is_x_down_edge = False

                # Extract the patch from latent data
                latent_patch = self.latent[:, :, :, :, :, patch_idx, :]

                # Process the patch # Only get enhanced image, pass seg
                t1 = time.time()
                enhanced_patch, enhanced_seg_patch = self.enhance_func(latent_patch,
                                                   augmentation=self.augmentation,
                                                   nx=nx,
                                                   is_x_up_edge=is_x_up_edge, is_x_down_edge=is_x_down_edge)
                # enhanced_patch = (enhanced_patch + 1) / 2
                # enhanced_seg_patch = (enhanced_seg_patch + 1) / 2
                # print(enhanced_patch.max(), enhanced_patch.min())
                # print(enhanced_seg_patch.max(), enhanced_seg_patch.min())
                # Ensure enhanced_patch is a numpy array for the signal
                if isinstance(enhanced_patch, torch.Tensor):
                    enhanced_patch = enhanced_patch.detach().cpu().numpy()
                if isinstance(enhanced_seg_patch, torch.Tensor):
                    enhanced_seg_patch = enhanced_seg_patch.detach().cpu().numpy()
                t2 = time.time()
                print("one yz time: ", t2-t1)

                # Emit signal with patch_idx and processed result
                self.patch_finished.emit(patch_idx, enhanced_patch[0, ::], enhanced_patch[1, ::], self.nz_start, self.nz_end, self.ny_start,
                                         self.ny_end, is_x_up_edge, is_x_down_edge)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class VisualizationBase(QWidget):
    def __init__(self, viewer, yaml_page, image_processer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.yaml_page = yaml_page
        self.image_processer = image_processer
        self.args = {}

        self.z_start = None
        self.z_end = None

    def instantiate_layout(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        self.register_model_button = QPushButton("Register Model")
        self.register_model_button.setFixedWidth(250)
        self.register_model_button.clicked.connect(self.register_model)
        self.register_model_button.installEventFilter(self)
        layout.addWidget(self.register_model_button)

        self.load_image_button = QPushButton("Load 3D Image")
        self.load_image_button.setFixedWidth(250)
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.load_latent_button = QPushButton("Load Latent")
        self.load_latent_button.setFixedWidth(250)
        self.load_latent_button.clicked.connect(self.load_latent)
        layout.addWidget(self.load_latent_button)

        self.show_boundary_button = QPushButton("Show Boundary")
        self.show_boundary_button.setFixedWidth(250)
        self.show_boundary_button.clicked.connect(self.show_boundary)
        layout.addWidget(self.show_boundary_button)

        self.manual_xy_button = QPushButton("Manual XY Boundary")
        self.manual_xy_button.setFixedWidth(250)
        self.manual_xy_button.clicked.connect(self.draw_manual_xy_boundary)
        layout.addWidget(self.manual_xy_button)

        self.enhance_image_button = QPushButton("Enhance Image")
        self.enhance_image_button.setFixedWidth(250)
        self.enhance_image_button.clicked.connect(self.enhance_image)
        self.enhance_image_button.installEventFilter(self)
        layout.addWidget(self.enhance_image_button)

        self.confirm_z_start_button = QPushButton("Confirm Z Start")
        self.confirm_z_start_button.setFixedWidth(250)
        self.confirm_z_start_button.clicked.connect(self.set_z_start)
        self.confirm_z_start_button.setVisible(False)
        layout.addWidget(self.confirm_z_start_button)

        self.confirm_z_end_button = QPushButton("Confirm Z End")
        self.confirm_z_end_button.setFixedWidth(250)
        self.confirm_z_end_button.clicked.connect(self.set_z_end)
        self.confirm_z_end_button.setVisible(False)
        layout.addWidget(self.confirm_z_end_button)

        self.confirm_manual_button = QPushButton("Confirm Manual Boundary")
        self.confirm_manual_button.setFixedWidth(250)
        self.confirm_manual_button.clicked.connect(self.confirm_manual_boundary)
        self.confirm_manual_button.setVisible(False)
        layout.addWidget(self.confirm_manual_button)

        self.viewer.dims.events.current_step.connect(self.on_z_slice_change)

        self.setLayout(layout)

        self.update_buttons_state()

    def eventFilter(self, obj, event):
        if obj is self.enhance_image_button and event.type() == QEvent.ToolTip:
            QToolTip.showText(event.globalPos(), self.enhance_image_button.toolTip())
            return True
        return super().eventFilter(obj, event)

    def update_buttons_state(self):
        def set_buttons_state(buttons, enabled, tooltip):
            for btn in buttons:
                btn.setEnabled(enabled)
                btn.setToolTip(tooltip)

        if not self.yaml_page.yaml_config:
            set_buttons_state(
                [self.load_image_button, self.load_latent_button, self.show_boundary_button,
                 self.manual_xy_button, self.enhance_image_button, self.register_model_button],
                False, "Please update YAML first!"
            )
        else:
            set_buttons_state([self.load_latent_button], True, "")

            if self.args:
                set_buttons_state(
                    [self.load_image_button, self.show_boundary_button, self.manual_xy_button,
                     self.register_model_button],
                    True, ""
                )
            else:
                set_buttons_state(
                    [self.load_image_button, self.show_boundary_button, self.manual_xy_button,
                     self.register_model_button],
                    False, "Please update args first!"
                )
            if self.args and self.registered_model and self.latent_loaded:
                set_buttons_state([self.enhance_image_button], True, "")
            else:
                set_buttons_state(
                    [self.enhance_image_button],
                    False, "Please register model and load latent first!"
                )


    def _update_params_to_processer(self):
        self.image_processer.args = self.args
        self.image_processer.kwargs = self.image_processer.process_config(self.yaml_page.yaml_config,
                                                                          self.image_processer.args.option)
        # This is dummy
        self.image_processer.save_image_datatype = self.image_processer.args.image_datatype
        if self.image_processer.normalization is None:
            self.image_processer.normalization = DataNormalization(
                backward_type=self.image_processer.save_image_datatype)


    def on_z_slice_change(self, event):
        if self.viewer.dims.ndisplay == 2:
            current_z = self.viewer.dims.current_step[0]
            if self.z_start is None:
                assemble_params = self.yaml_page.yaml_config.get("DEFAULT", {}).get("assemble_params", {})
                dx_shape = assemble_params.get("dx_shape", None)
                xrange_val = assemble_params.get("xrange", None)
                if dx_shape is not None and xrange_val is not None:
                    self.z_start = xrange_val[0]
                    self.z_end = xrange_val[1] + dx_shape[1]
            if "Boundary" in self.viewer.layers:
                self.viewer.layers["Boundary"].visible = (self.z_start <= current_z <= self.z_end)


    def draw_manual_xy_boundary(self):
        if "Manual XY Boundary" in self.viewer.layers:
            self.viewer.layers.remove("Manual XY Boundary")
        self.viewer.add_shapes(np.empty((0, 0, 2)), shape_type="polygon", name="Manual XY Boundary",
                               face_color="transparent", edge_color="yellow", edge_width=2)
        self.confirm_z_start_button.setVisible(True)
        self.confirm_z_end_button.setVisible(False)
        self.confirm_manual_button.setVisible(False)


    def set_z_start(self):
        z_val = self.viewer.dims.current_step[0]
        self.z_start = z_val
        print("Z Start set to:", z_val)
        self.confirm_z_end_button.setVisible(True)


    def set_z_end(self):
        z_val = self.viewer.dims.current_step[0]
        self.z_end = z_val
        print("Z End set to:", z_val)
        self.confirm_z_end_button.setVisible(False)
        self.confirm_manual_button.setVisible(True)


    def update_enhanced_zarr(self, patch_idx, enhanced_patch, enhanced_seg_patch, nz_start, nz_end, ny_start, ny_end, is_x_up_edge,
                             is_x_down_edge):
        # 計算該 patch 在 x 軸上的起始與結束位置
        moving_size_x = self.image_processer.kwargs["assemble_params"]["xrange"][2]
        moving_size_z = self.image_processer.kwargs["assemble_params"]["zrange"][2] * self.image_processer.kwargs[
            "N_resolution"]
        moving_size_y = self.image_processer.kwargs["assemble_params"]["yrange"][2]

        patch_size_z, patch_size_x, patch_size_y = self.image_processer.kwargs["assemble_params"]["dx_shape"]
        patch_size_z = patch_size_z * self.image_processer.kwargs["N_resolution"]
        C0, C1, C2 = self.image_processer.kwargs["assemble_params"]["C"]
        S0, S1, S2 = self.image_processer.kwargs["assemble_params"]["S"]

        region_z_start = nz_start * moving_size_z
        region_z_end = (nz_end - 1) * moving_size_z + patch_size_z

        region_y_start = ny_start * moving_size_y
        region_y_end = (ny_end - 1) * moving_size_y + patch_size_y

        if is_x_up_edge:
            region_x_start = patch_idx * moving_size_x
        else:
            region_x_start = patch_idx * moving_size_x + C1

        if is_x_down_edge:
            region_x_end = patch_idx * moving_size_x + patch_size_x
        else:
            region_x_end = patch_idx * moving_size_x + patch_size_x - C1

        if not is_x_up_edge:
            # if is_x_down_edge:
            import tifffile as tiff
            tiff.imwrite("before.tif", enhanced_patch)
            enhanced_patch[:S1, C0:-C0, C2:-C2] += self.enhanced_layer.data[region_x_start:region_x_start + S1,
                                                   region_z_start + C0:region_z_end - C0,
                                                   region_y_start + C2:region_y_end - C2]
            tiff.imwrite("after.tif", enhanced_patch)
            enhanced_seg_patch[:S1, C0:-C0, C2:-C2] += self.enhanced_seg_layer.data[region_x_start:region_x_start + S1,
                                                   region_z_start + C0:region_z_end - C0,
                                                   region_y_start + C2:region_y_end - C2]
            # else:
            #     enhanced_patch[:S1, C0:-C0, C2:-C2] += self.enhanced_layer.data[region_x_start:region_x_start + S1,
            #                                            region_z_start + C0:region_z_end - C0,
            #                                            region_y_start + C2:region_y_end - C2]

        # Get the ROI coordinates from YAML config
        default = self.yaml_page.yaml_config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        xrange_val = assemble_params.get("xrange", None)
        zrange_val = assemble_params.get("zrange", None)
        yrange_val = assemble_params.get("yrange", None)
        dx_shape = assemble_params.get("dx_shape", None)

        if xrange_val is not None and zrange_val is not None and yrange_val is not None and dx_shape is not None:
            # Get the N_resolution factor for z-axis upsampling
            N_resolution = self.image_processer.kwargs["N_resolution"]

            # Calculate the desired ROI boundaries
            x_start_roi = xrange_val[0]
            x_end_roi = xrange_val[1] + dx_shape[1]
            z_start_roi = zrange_val[0] * N_resolution
            z_end_roi = (zrange_val[1] + dx_shape[0]) * N_resolution
            y_start_roi = yrange_val[0]
            y_end_roi = yrange_val[1] + dx_shape[2]

            x_start_offset = max(x_start_roi - region_x_start, 0)
            x_end_offset = min(enhanced_patch.shape[0], x_end_roi - region_x_start)
            z_start_offset = z_start_roi - region_z_start
            z_end_offset = z_end_roi - region_z_end
            y_start_offset = y_start_roi - region_y_start
            y_end_offset = y_end_roi - region_y_end
            x_slice = slice(x_start_offset, None) if x_end_offset == 0 else slice(x_start_offset, x_end_offset)
            z_slice = slice(z_start_offset, None) if z_end_offset == 0 else slice(z_start_offset, z_end_offset)
            y_slice = slice(y_start_offset, None) if y_end_offset == 0 else slice(y_start_offset, y_end_offset)
            print(
                f"update enhance layer : {max(x_start_roi - region_x_start, region_x_start)}, {region_x_start + x_end_offset}, {z_start_roi}, {z_end_roi}, {y_start_roi}, {y_end_roi}")
            print(f"enhance patch shape: {enhanced_patch.shape}, slice enahnce patch: {x_slice} {z_slice} {y_slice}")
            enhanced_region_start = max(x_start_roi, region_x_start)
            enhanced_region_end = region_x_start + x_end_offset
            self.enhanced_layer.data[enhanced_region_start:enhanced_region_end,
            z_start_roi:z_end_roi,
            y_start_roi:y_end_roi] = enhanced_patch[x_slice, z_slice, y_slice]
            self.enhanced_seg_layer.data[enhanced_region_start:enhanced_region_end,
            z_start_roi:z_end_roi,
            y_start_roi:y_end_roi] = enhanced_seg_patch[x_slice, z_slice, y_slice]


        self.enhanced_layer.refresh()  # 更新顯示
        self.enhanced_seg_layer.refresh() # 更新顯示

    def iter_enhance_by_yz_slice(self, yz_latent, augmentation=[None], nx=-1, is_x_up_edge=True, is_x_down_edge=True):
        """
        Process one yz axis volume.
        Assumes shape is [f_z, 4, f_x, f_y, idx_z, idx_y]
        """
        print("iter_enhance_by_yz_slice start ")
        C0, C1, C2 = self.image_processer.kwargs['assemble_params']['C']
        S0, S1, S2 = self.image_processer.kwargs['assemble_params']['S']

        # Check if input is on GPU
        device = yz_latent.device if isinstance(yz_latent, torch.Tensor) else 'cpu'

        one_column = []
        one_seg_column = []
        one_column_coord = []
        for num_z, nz in enumerate(range(yz_latent.shape[4])):
            one_row = []
            one_seg_row = []
            one_row_coord = []
            # Determine z-axis edge conditions
            if yz_latent.shape[4] == 1:
                is_z_up_edge = True
                is_z_down_edge = True
                nz = -2
            elif nz == 0:
                is_z_up_edge = True
                is_z_down_edge = False
            elif nz == yz_latent.shape[4] - 1:
                nz = -1
                is_z_up_edge = False
                is_z_down_edge = True
            else:
                is_z_up_edge = False
                is_z_down_edge = False

            for num_y, ny in enumerate(range(yz_latent.shape[5])):
                # Determine y-axis edge conditions
                if yz_latent.shape[5] == 1:
                    is_y_up_edge = True
                    is_y_down_edge = True
                    ny = -2
                elif ny == 0:
                    is_y_up_edge = True
                    is_y_down_edge = False
                elif ny == yz_latent.shape[5] - 1:
                    ny = -1
                    is_y_up_edge = False
                    is_y_down_edge = True
                else:
                    is_y_up_edge = False
                    is_y_down_edge = False

                # Create tapered weight for blending
                w = create_tapered_weight(S0, S1, S2, nz, nx, ny,
                                          size=self.image_processer.kwargs['assemble_params']['weight_shape'],
                                          edge_size=64)

                # Process the patch
                out_all, out_all_seg = self.image_processer.test_ae_decode(yz_latent[:, :, :, :, num_z, num_y], augmentation)
                out_all = (out_all + 1) / 2
                # Create output tensor on the same device as input
                # out_all = torch.ones((256, 2, 256, 256), device=device)

                # Apply edge trimming based on edge conditions
                if not is_z_up_edge:
                    out_all = out_all[C0:, ::]
                    out_all_seg = out_all_seg[C0:, ::]
                    multi_z_coord_start = 0
                else:
                    multi_z_coord_start = C0
                if not is_z_down_edge:
                    out_all = out_all[:-C0, ::]
                    out_all_seg = out_all_seg[:-C0, ::]
                    multi_z_coord_end = out_all.shape[0]
                else:
                    multi_z_coord_end = -C0

                if not is_x_up_edge:
                    out_all = out_all[:, :, C1:, :]
                    out_all_seg = out_all_seg[:, :, C1:, :]
                    multi_x_coord_start = 0
                else:
                    multi_x_coord_start = C1
                if not is_x_down_edge:
                    out_all = out_all[:, :, :-C1, :]
                    out_all_seg = out_all_seg[:, :, :-C1, :]
                    multi_x_coord_end = out_all.shape[2]
                else:
                    multi_x_coord_end = -C1

                if not is_y_up_edge:
                    out_all = out_all[:, :, :, C2:]
                    out_all_seg = out_all_seg[:, :, :, C2:]
                    multi_y_coord_start = 0
                else:
                    multi_y_coord_start = C2
                if not is_y_down_edge:
                    out_all = out_all[:, :, :, :-C2]
                    out_all_seg = out_all_seg[:, :, :, :-C2]
                    multi_y_coord_end = out_all.shape[3]
                else:
                    multi_y_coord_end = -C2

                # Move to CPU for numpy operations if needed
                if device != 'cpu':
                    out_all_cpu = out_all.detach().cpu()
                    out_all_seg = out_all_seg.detach().cpu()
                else:
                    out_all_cpu = out_all.detach()
                    out_all_seg = out_all_seg.detach()

                # Apply weight to the patch
                w = np.stack([w] * out_all_cpu.shape[1], axis=1)
                out_all_cpu[multi_z_coord_start:multi_z_coord_end, :, multi_x_coord_start:multi_x_coord_end, multi_y_coord_start:multi_y_coord_end] = (
                    np.multiply(out_all_cpu[
                        multi_z_coord_start:multi_z_coord_end,
                        :,
                        multi_x_coord_start:multi_x_coord_end,
                        multi_y_coord_start:multi_y_coord_end
                    ], w))
                out_all_seg[multi_z_coord_start:multi_z_coord_end, :, multi_x_coord_start:multi_x_coord_end, multi_y_coord_start:multi_y_coord_end] = (
                    np.multiply(out_all_seg[
                        multi_z_coord_start:multi_z_coord_end,
                        :,
                        multi_x_coord_start:multi_x_coord_end,
                        multi_y_coord_start:multi_y_coord_end
                    ], w))

                # Handle blending with previous row
                if len(one_row) > 0:
                    last_row_coord = one_row_coord[-1]
                    one_row[-1][last_row_coord[0]:last_row_coord[1], :, last_row_coord[2]:last_row_coord[3],
                    -S2:] += out_all_cpu[multi_z_coord_start:multi_z_coord_end, :,
                             multi_x_coord_start:multi_x_coord_end, :S2]
                    one_seg_row[-1][last_row_coord[0]:last_row_coord[1], :, last_row_coord[2]:last_row_coord[3],
                    -S2:] += out_all_seg[multi_z_coord_start:multi_z_coord_end, :,
                             multi_x_coord_start:multi_x_coord_end, :S2]
                    # import tifffile as tiff
                    # tiff.imwrite(f"nz_nx_ny_{nz}_{nx}_{ny}.tif", one_row[-1].detach().numpy().astype(np.float32))
                    one_row.append(out_all_cpu[:, :, :, S2:])
                    one_seg_row.append(out_all_seg[:, :, :, S2:])
                else:
                    one_row.append(out_all_cpu)
                    one_seg_row.append(out_all_seg)
                one_row_coord.append((multi_z_coord_start, multi_z_coord_end, multi_x_coord_start, multi_x_coord_end,
                                      multi_y_coord_start, multi_y_coord_end))

            # Concatenate rows
            one_row = np.concatenate(one_row, axis=3)  # (Z, C, X, Y)
            one_row = np.transpose(one_row, (1, 2, 0, 3))  # (C, X, Z, Y)
            one_seg_row = np.concatenate(one_seg_row, axis=3)  # (Z, C, X, Y)
            one_seg_row = np.transpose(one_seg_row, (1, 2, 0, 3))  # (C, X, Z, Y)

            if len(one_column) > 0:
                last_column_coord = one_column_coord[-1]
                one_column[-1][:, last_column_coord[0]:last_column_coord[1], -S0:, C2:-C2] += one_row[:,
                                                                                              multi_x_coord_start:multi_x_coord_end,
                                                                                              :S0, C2:-C2]
                one_seg_column[-1][:, last_column_coord[0]:last_column_coord[1], -S0:, C2:-C2] += one_seg_row[:,
                                                                                              multi_x_coord_start:multi_x_coord_end,
                                                                                              :S0, C2:-C2]
                one_column.append(one_row[:, :, S0:, :])
                one_seg_column.append(one_seg_row[:, :, S0:, :])
            else:
                one_column.append(one_row)
                one_seg_column.append(one_seg_row)

            one_column_coord.append((multi_x_coord_start, multi_x_coord_end))

        # Concatenate columns
        one_column = np.concatenate(one_column, axis=2).astype(np.float32)  # (C, X, Z, Y)
        one_seg_column = np.concatenate(one_seg_column, axis=2).astype(np.float32)  # (C, X, Z, Y)


        return one_column, one_seg_column

    def get_patch_index(self, coordinate_start, coordinate_end, patch_size, moving_size):
        """
        Given a coordinate, patch_size, and moving_size,
        return the patch index for the start coordinate (floor) and end coordinate (ceil)
        """
        start_index = int(math.floor(coordinate_start / moving_size))
        offset = coordinate_end - start_index * moving_size
        if offset <= patch_size:
            end_index = start_index + 1
        else:
            end_index = start_index + int(math.ceil((offset - patch_size) / moving_size)) + 1
        # return int(math.floor(coordinate_start / moving_size)), max(0, int(math.ceil(
        #     (coordinate_end - patch_size) / moving_size)) + 1)
        return start_index, end_index

    def calculate_is_need_enhance_region_index(self, xrange_val, zrange_val, yrange_val):
        default = self.yaml_page.yaml_config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        dz, dx, dy = assemble_params.get("dx_shape", None)
        # This is coord for entire marked region, maybe will need larger when enhanced
        x_start, x_end = xrange_val[0], xrange_val[1] + dx
        y_start, y_end = yrange_val[0], yrange_val[1] + dy
        z_start, z_end = zrange_val[0], zrange_val[1] + dz
        # Now calculate index
        index_x_start, index_x_end = self.get_patch_index(x_start, x_end, dx, xrange_val[2])
        index_y_start, index_y_end = self.get_patch_index(y_start, y_end, dy, yrange_val[2])
        index_z_start, index_z_end = self.get_patch_index(z_start, z_end, dz, zrange_val[2])
        return index_x_start, index_x_end, index_z_start, index_z_end, index_y_start, index_y_end

    def on_model_loaded(self, model):
        self.registered_model = True
        self.register_model_button.setEnabled(True)
        self.update_buttons_state()
        print("Model loaded successfully.")

    def on_model_error(self, error_msg):
        self.register_model_button.setEnabled(True)
        print("Error loading model:", error_msg)

    def show_new_viewer(self):
        if not hasattr(self, 'lazy_zarr') or self.lazy_zarr is None:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please Load Images first"
            )
            return
        if not hasattr(self, 'enhanced_zarr') or self.lazy_zarr is None:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please Load Images first"
            )
            return
        config = self.yaml_page.yaml_config
        default = config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        dx_shape = assemble_params.get("dx_shape", None)
        xrange_val = assemble_params.get("xrange", None)
        zrange_val = assemble_params.get("zrange", None)
        yrange = assemble_params.get("yrange", None)

        x0 = zrange_val[0] #* self.image_processer.kwargs["N_resolution"]
        x1 = (zrange_val[1] + dx_shape[0])# * self.image_processer.kwargs["N_resolution"]
        y0 = yrange[0]
        y1 = yrange[1] + dx_shape[2]
        z0 = xrange_val[0]

        upsample_factor = self.image_processer.kwargs["N_resolution"]
        shift_world = 0.5 * upsample_factor
        viewer = napari.Viewer()
        viewer.add_image(
            self.lazy_zarr[z0:z0+32, x0:x1, y0:y1],
            name="3D Image",
            scale=(1, upsample_factor, 1),
            translate=(0, shift_world, 0),
            interpolation="linear"
        )
        viewer.add_image(
            self.enhanced_zarr[z0:z0+32, x0*upsample_factor:x1*upsample_factor, y0:y1],
            name="Enhanced Image",
            scale=(1, 1, 1)
        )

    def run_fid(self):
        if not hasattr(self, 'lazy_zarr') or self.lazy_zarr is None:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please Load Images first"
            )
            return
        if not hasattr(self, 'enhanced_zarr') or self.lazy_zarr is None:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please Load Images first"
            )
            return
        config = self.yaml_page.yaml_config
        default = config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        dx_shape = assemble_params.get("dx_shape", None)
        xrange_val = assemble_params.get("xrange", None)
        zrange_val = assemble_params.get("zrange", None)
        yrange = assemble_params.get("yrange", None)

        x0 = zrange_val[0] #* self.image_processer.kwargs["N_resolution"]
        x1 = (zrange_val[1] + dx_shape[0])# * self.image_processer.kwargs["N_resolution"]
        y0 = yrange[0]
        y1 = yrange[1] + dx_shape[2]
        z0 = xrange_val[0]

        upsample_factor = self.image_processer.kwargs["N_resolution"]
        real_xy = np.transpose(self.lazy_zarr[:, x0:x1, :], (1, 0, 2))
        ori_image = self.lazy_zarr[z0:z0 + 32, x0:x1, y0:y1]

        from skimage.transform import resize
        ori_image = resize(ori_image,
                         (ori_image.shape[0],
                          ori_image.shape[1] * upsample_factor,
                          ori_image.shape[2]),
                         order=1,
                         preserve_range=True,
                         anti_aliasing=False)

        enhance_image = self.enhanced_zarr[z0:z0+32, x0*upsample_factor:x1*upsample_factor, y0:y1]
        from utils.metrics.metrics import FID3DCalculator
        calculator = FID3DCalculator(device='cuda', batch_size=4)
        print("fid calculator init ok")
        ori_fid = calculator.calculate_fid(real_vols=real_xy, fake_vols=ori_image)
        print("ori fid: ", ori_fid)
        enhanced_fid = calculator.calculate_fid(real_vols=real_xy, fake_vols=enhance_image)
        print("enhanced fid: ", enhanced_fid)
        QMessageBox.information(self, "3D FID", f"Before Enhance FID = {ori_fid:.4f}\nAfter Enhanced FID = {enhanced_fid: .4f}")