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
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton, QFileDialog,
    QHBoxLayout, QPlainTextEdit, QFormLayout, QLineEdit, QCheckBox, QComboBox, QApplication
)
from qtpy.QtCore import Qt, Signal, QThread, QEvent
from qtpy.QtWidgets import QToolTip, QScrollArea

from test_assemble import MicroTest
from utils.base_micro_test import create_tapered_weight
from utils.data_utils import _CHECK_PARAMS, DataNormalization
import tifffile as tiff

from napari.utils.colormaps import Colormap
from utils.napari_utils import VisualizationBase, ModelLoaderThread, UpsampleCopyWorker, EnhanceWorker


# Keep yaml same format
yaml_ruamel = YAML()
yaml_ruamel.preserve_quotes = True


class ArgsPanel(QWidget):
    args_updated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--option', type=str, default="VMAT", help='which dataset to use')
        self.parser.add_argument('--mc', type=str, default="1", help='monte carlo inference, mean over N times')
        self.parser.add_argument('--testpatch', action='store_true', default=False)
        self.parser.add_argument('--testcube', action='store_true', default=False)
        self.parser.add_argument('--gpu', action='store_true', default=False)
        self.parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 inference')
        self.parser.add_argument('--save', nargs='+', choices=['ori', 'recon', 'xy'], required=False)
        self.parser.add_argument('--image_datatype', type=str, choices=["float32", "float16", "uint16", "uint8"], default="float32")
        self.parser.add_argument('--augmentation', type=str, default="decode")
        self.parser.add_argument('--augmentation_method', nargs='*', default=[], help='Augmentation methods to apply: None, transpose, flipX, flipY')
        self.parser.add_argument('--reslice', action='store_true', default=False)
        self.parser.add_argument('--host', type=str, default='dummy')
        self.parser.add_argument('--port', type=str, default='dummy')
        self.parser.add_argument('--assemble_method', type=str, default='tiff', help='tiff or zarr method while assemble images')
        self.parser.add_argument('--roi', type=str, default='')
        self.parser.add_argument('--targets', nargs='+', default=None, required=False, help="assign target to assemble")

        self.args = self.parser.parse_args([])

        layout = QFormLayout()
        self.widgets = {}
        self.gpu_checkboxes = {}  # 儲存 GPU 勾選框

        for action in self.parser._actions:
            if action.dest == 'help':
                continue

            if action.dest == "option":
                widget = QComboBox()
            elif action.choices is not None and action.nargs is None:
                widget = QComboBox()
                for choice in action.choices:
                    widget.addItem(str(choice))
            elif isinstance(action.default, bool):
                widget = QCheckBox()
                widget.setChecked(action.default)

                # 特別處理 --gpu：當勾選時顯示 GPU 列表
                if action.dest == "gpu":
                    widget.stateChanged.connect(self.on_gpu_checked)
                    # 加一個容器來放 GPU 勾選框（水平排列）
                    self.gpu_checkbox_container = QWidget()
                    gpu_layout = QHBoxLayout()
                    gpu_layout.setContentsMargins(0, 0, 0, 0)
                    self.gpu_checkbox_container.setLayout(gpu_layout)
                    self.gpu_checkbox_container.setVisible(False)
            elif action.dest == "augmentation_method":
                widget = QWidget()
                h_layout = QHBoxLayout()
                h_layout.setContentsMargins(0, 0, 0, 0)
                checkboxes = {}
                for option in ["None", "transpose", "flipX", "flipY"]:
                    checkbox = QCheckBox(option)
                    checkbox.setChecked(option in action.default)
                    h_layout.addWidget(checkbox)
                    checkboxes[option] = checkbox
                h_layout.addStretch()
                widget.setLayout(h_layout)
                widget.checkboxes = checkboxes
            elif action.nargs in ['+', '*']:
                default_text = ", ".join(action.default) if action.default else ""
                widget = QLineEdit(default_text)
            else:
                widget = QLineEdit(str(action.default))

            layout.addRow(action.dest, widget)
            self.widgets[action.dest] = widget

            if action.dest == "gpu":
                layout.addRow("gpu_ids", self.gpu_checkbox_container)

        self.update_button = QPushButton("Update Args")
        self.update_button.clicked.connect(self.on_update)

        v_layout = QVBoxLayout()
        v_layout.addLayout(layout)
        v_layout.addWidget(self.update_button)
        self.setLayout(v_layout)

    def on_gpu_checked(self, state):
        if state == Qt.Checked:
            try:
                import torch
                num_gpus = torch.cuda.device_count()
                gpu_layout = self.gpu_checkbox_container.layout()
                while gpu_layout.count():
                    child = gpu_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                self.gpu_checkboxes.clear()
                for i in range(num_gpus):
                    name = torch.cuda.get_device_name(i)
                    checkbox = QCheckBox(f"GPU {i} ({name})")
                    checkbox.setChecked(True)
                    gpu_layout.addWidget(checkbox)
                    self.gpu_checkboxes[i] = checkbox
                self.gpu_checkbox_container.setVisible(True)
            except Exception as e:
                print(f"Error scanning GPU: {e}")
                self.gpu_checkbox_container.setVisible(False)
        else:
            self.gpu_checkbox_container.setVisible(False)

    def update_option_choices(self, yaml_config):
        if not isinstance(yaml_config, dict):
            return
        options = [key for key in yaml_config.keys() if key != "DEFAULT"]
        combo: QComboBox = self.widgets.get("option")
        if combo is None:
            return
        combo.clear()
        if options:
            combo.addItems(options)
        else:
            combo.addItem("")

    def on_update(self):
        for action in self.parser._actions:
            if action.dest == 'help':
                continue
            widget = self.widgets.get(action.dest)
            if widget is None:
                continue

            if isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif hasattr(widget, 'checkboxes'):
                value = [option for option, checkbox in widget.checkboxes.items() if checkbox.isChecked()]
                if "None" in value:
                    value.remove("None")
                    value = [None] + value
            elif hasattr(widget, "text"):
                text = widget.text()
                if action.nargs in ['+', '*']:
                    value = [s.strip() for s in text.split(",") if s.strip()] if text else []
                else:
                    try:
                        value = action.type(text)
                    except Exception as e:
                        print(f"Error converting {action.dest}: {e}")
                        value = text
            else:
                value = widget
            setattr(self.args, action.dest, value)

        # 另外加上 GPU ID 列表
        if hasattr(self, "gpu_checkboxes"):
            selected_gpus = [i for i, cb in self.gpu_checkboxes.items() if cb.isChecked()]
            setattr(self.args, "gpu_ids", selected_gpus)

        print("Updated args:", self.args)
        self.args_updated.emit(self.args)



class YAMLConfigPage(QWidget):
    # Signal emitted when YAML config is updated, passes dictionary data
    yaml_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.yaml = yaml_ruamel  # Use ruamel.yaml instance
        layout = QVBoxLayout()
        # Text editor to display YAML content
        self.editor = QPlainTextEdit()
        layout.addWidget(self.editor)

        # Load YAML and Update YAML buttons
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load YAML")
        self.load_button.clicked.connect(self.load_yaml)
        btn_layout.addWidget(self.load_button)

        self.update_button = QPushButton("Update YAML")
        self.update_button.clicked.connect(self.update_yaml)
        btn_layout.addWidget(self.update_button)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # Store parsed YAML settings
        self.yaml_config = {}

    def load_yaml(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load YAML File", "", "YAML Files (*.yaml *.yml)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            self.editor.setPlainText(content)
            # Parse with ruamel.yaml and preserve original format
            self.yaml_config = self.yaml.load(content)
            print(f"YAML file loaded: {filename}")
            # Emit signal to update YAML config
            self.yaml_updated.emit(self.yaml_config)

    def update_yaml(self):
        content = self.editor.toPlainText()
        try:
            self.yaml_config = self.yaml.load(content)
            # Emit signal to update YAML config
            self.yaml_updated.emit(self.yaml_config)
        except Exception as e:
            print("Error updating YAML:", e)


class AnalysisPage(QWidget):
    def __init__(self, visualization_page, parent=None):
        super().__init__(parent)
        self.visualization_page = visualization_page
        layout = QVBoxLayout()


        btn_new = QPushButton("Open New Viewer")
        btn_new.clicked.connect(self.visualization_page.show_new_viewer)
        layout.addWidget(btn_new)

        btn_calculate_fid_analysis = QPushButton("Calculate FID")
        btn_calculate_fid_analysis.clicked.connect(self.visualization_page.run_fid)
        layout.addWidget(btn_calculate_fid_analysis)

        layout.addStretch()
        self.setLayout(layout)



class VisualizationPage(VisualizationBase):
    def __init__(self, viewer, yaml_page, image_processer, parent=None):
        super().__init__(viewer, yaml_page, image_processer, parent)

        # Parameters used in this class
        self.registered_model = False
        self.latent_loaded = False
        self.instantiate_layout()

    def update_args(self, args):
        """Receive updated parameters from ArgsPanel and check button states"""
        self.args = args
        print("VisualizationPage updated with args:", self.args)
        self.update_buttons_state()

    def load_image(self):
        self._update_params_to_processer()

        # Let user select folder containing original lazy_zarr data
        folder = QFileDialog.getExistingDirectory(self, "Select Zarr Folder", os.getcwd())
        if not folder:
            print("No folder selected!")
            return

        # Open original lazy_zarr (shape assumed to be (x, z, y))
        self.lazy_zarr = zarr.open(folder, mode='r')
        print("Lazy zarr shape:", self.lazy_zarr.shape)

        # Get upsample_factor from processer's kwargs
        upsample_factor = self.image_processer.kwargs["N_resolution"]

        # Create enhanced_zarr based on original lazy_zarr's shape and chunks
        (x, z, y) = self.lazy_zarr.shape
        # Target shape: only z axis expanded by upsample_factor
        new_shape = (x, z * upsample_factor, y)
        # new_chunks = (x_chunk, z_chunk * upsample_factor, y_chunk)
        # enhanced_store_path = os.path.join(os.getcwd(), "temp_enhanced.zarr")
        # enhanced_zarr = zarr.open(enhanced_store_path, mode='r+')
        # if os.path.exists(enhanced_store_path):
        #     shutil.rmtree(enhanced_store_path)
        # enhanced_zarr = zarr.open(enhanced_store_path, mode='w',
        #                           shape=new_shape,
        #                           chunks=new_chunks,
        #                           dtype=self.lazy_zarr.dtype)
        # # 開始在背景線程中分塊複製原始 lazy_zarr 到 enhanced_zarr（同時上採樣 z 軸）
        #
        # self.copy_worker = UpsampleCopyWorker(self.lazy_zarr, enhanced_zarr, upsample_factor=upsample_factor)
        # self.copy_worker.progress.connect(lambda s: print("Copied chunk at", s))
        # self.copy_worker.finished.connect(lambda: print("Zarr copying finished."))
        # self.copy_worker.error.connect(lambda err: print("Error copying zarr:", err))
        # self.copy_worker.start()

        # self.enhanced_zarr = enhanced_zarr

        self.enhanced_zarr = zarr.open(None, mode='w', shape=new_shape,
                                       dtype=np.float32, chunks=(128, 256, 256), fill_value=np.nan)

        self.enhanced_seg_zarr = zarr.open(None, mode='w', shape=new_shape,
                                       dtype=np.float32, chunks=(128, 256, 256), fill_value=np.nan)

        # Add original lazy_zarr to napari view
        # Correct shift
        shift_world = 0.5 * upsample_factor
        self.viewer.add_image(self.lazy_zarr, name="3D Image", scale=(1, upsample_factor, 1), interpolation="linear",
                              translate=(0, shift_world, 0))
        self.enhanced_layer = self.viewer.add_image(self.enhanced_zarr, name="Enhanced Image", scale=(1, 1, 1))# , rgb=False, colormap='gray'
        self.enhanced_seg_layer = self.viewer.add_image(self.enhanced_seg_zarr, name="Enhanced Seg", scale=(1, 1, 1),
                                                        colormap='magma', contrast_limits=[0, 1], blending='additive')

        self.enhanced_layer.opacity = 1.0
        self.enhanced_seg_layer.opacity = 1.0
        # self.enhanced_layer.opacity = np.zeros_like(self.enhanced_layer.data)

        # Also add enhanced_zarr to napari view as Enhanced Image layer
        # This layer's scale is set to (1,1,1) since the data itself is already upsampled

        print("Loaded image layers.")
        self.update_buttons_state()

    def load_latent(self):
        self._update_params_to_processer()
        self.latent = self.image_processer.get_data()
        print(f"Loaded latent!")
        self.latent_loaded = True
        self.update_buttons_state()

    def show_boundary(self):
        config = self.yaml_page.yaml_config
        try:
            default = config.get("DEFAULT", {})
            assemble_params = default.get("assemble_params", {})
            dx_shape = assemble_params.get("dx_shape", None)
            xrange_val = assemble_params.get("xrange", None)
            zrange_val = assemble_params.get("zrange", None)
            yrange = assemble_params.get("yrange", None)
            if dx_shape is None or xrange_val is None or zrange_val is None or yrange is None:
                print("assemble_params data incomplete!")
                return

            boundary_color = assemble_params.get("boundary_color", "white")
            boundary_line_width = assemble_params.get("boundary_line_width", 4)
            face_color = "transparent"

            if self.viewer.dims.ndisplay == 2:
                self.z_start = xrange_val[0]
                self.z_end = xrange_val[1] + dx_shape[1] - 1

                x0 = zrange_val[0] * self.image_processer.kwargs["N_resolution"]
                x1 = (zrange_val[1] + dx_shape[0]) * self.image_processer.kwargs["N_resolution"]
                y0 = yrange[0]
                y1 = yrange[1] + dx_shape[2]
                shape_data = np.array([[x0, y0],
                                       [x1, y0],
                                       [x1, y1],
                                       [x0, y1],
                                       [x0, y0]])
                shape_data = shape_data[None, ...]
                shape_type = "polygon"
            else:
                dx_shape = list(dx_shape)
                z0 = xrange_val[0]
                z1 = xrange_val[1] + dx_shape[0]
                x0 = zrange_val[0] * self.image_processer.kwargs["N_resolution"]
                x1 = (zrange_val[1] + dx_shape[0]) * self.image_processer.kwargs["N_resolution"]
                y0 = yrange[0]
                y1 = yrange[1] + dx_shape[2]
                edges = [
                    [[z0, x0, y0], [z0, x0, y1]],
                    [[z0, x0, y1], [z0, x1, y1]],
                    [[z0, x1, y1], [z0, x1, y0]],
                    [[z0, x1, y0], [z0, x0, y0]],
                    [[z1, x0, y0], [z1, x0, y1]],
                    [[z1, x0, y1], [z1, x1, y1]],
                    [[z1, x1, y1], [z1, x1, y0]],
                    [[z1, x1, y0], [z1, x0, y0]],
                    [[z0, x0, y0], [z1, x0, y0]],
                    [[z0, x0, y1], [z1, x0, y1]],
                    [[z0, x1, y1], [z1, x1, y1]],
                    [[z0, x1, y0], [z1, x1, y0]],
                ]
                shape_data = [np.array(edge, dtype=np.float32) for edge in edges]
                shape_type = "line"

            if "Boundary" in self.viewer.layers:
                self.viewer.layers.remove("Boundary")
            self.viewer.add_shapes(shape_data, shape_type=shape_type, name="Boundary",
                                   edge_color=boundary_color, edge_width=boundary_line_width,
                                   face_color=face_color)
            print("Boundary displayed.")
        except Exception as e:
            print("Error in show_boundary:", e)

    def confirm_manual_boundary(self):
        if "Manual XY Boundary" not in self.viewer.layers:
            print("No XY boundary drawn yet.")
            return
        xy_layer = self.viewer.layers["Manual XY Boundary"]
        if not xy_layer.data or len(xy_layer.data) == 0:
            print("No XY boundary shape found.")
            return
        xy_shape = np.array(xy_layer.data[0])
        x_coords = xy_shape[:, 0]
        y_coords = xy_shape[:, 1]
        x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
        y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
        if self.z_start is None or self.z_end is None:
            print("Please confirm Z axis start and end positions first.")
            return
        config = self.yaml_page.yaml_config
        default = config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        dx_shape = assemble_params.get("dx_shape", None)
        if dx_shape is None:
            print("YAML missing dx_shape, cannot update boundary parameters.")
            return
        new_xrange = [int(self.z_start), int(self.z_end - dx_shape[1])]
        new_zrange = [int(x_min / self.image_processer.kwargs["N_resolution"]),
                      int(x_max / self.image_processer.kwargs["N_resolution"] - dx_shape[0])]
        new_yrange = [int(y_min), int(y_max - dx_shape[2])]
        new_xrange_cs = CommentedSeq(new_xrange)
        new_xrange_cs.fa.set_flow_style()
        new_zrange_cs = CommentedSeq(new_zrange)
        new_zrange_cs.fa.set_flow_style()
        new_yrange_cs = CommentedSeq(new_yrange)
        new_yrange_cs.fa.set_flow_style()

        assemble_params["xrange"] = new_xrange_cs
        assemble_params["zrange"] = new_zrange_cs
        assemble_params["yrange"] = new_yrange_cs
        default["assemble_params"] = assemble_params
        config["DEFAULT"] = default

        s = StringIO()
        self.yaml_page.yaml.dump(config, s)
        new_yaml_text = s.getvalue()
        self.yaml_page.editor.setPlainText(new_yaml_text)
        self.yaml_page.yaml_config = config
        self.confirm_z_start_button.setVisible(False)
        self.confirm_z_end_button.setVisible(False)
        self.confirm_manual_button.setVisible(False)

        print("Manual boundary updated in YAML:")
        print("xrange:", new_xrange, "zrange:", new_zrange, "yrange:", new_yrange)
        if "Manual XY Boundary" in self.viewer.layers:
            self.viewer.layers.remove("Manual XY Boundary")
        self.show_boundary()

    def enhance_image(self):
        self._update_params_to_processer()
        default = self.yaml_page.yaml_config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        xrange_val = assemble_params.get("xrange", None)
        zrange_val = assemble_params.get("zrange", None)
        yrange_val = assemble_params.get("yrange", None)
        dz, dx, dy = assemble_params.get("dx_shape", None)
        S0, S1, S2 = self.image_processer.kwargs["assemble_params"]["S"]
        print("Enhance Image - Current Parameters:")
        print("xrange:", xrange_val)
        print("zrange:", zrange_val)
        print("yrange:", yrange_val)

        index_x_start, index_x_end, index_z_start, index_z_end, index_y_start, index_y_end = (
            self.calculate_is_need_enhance_region_index(xrange_val, zrange_val, yrange_val))
        print("index_x_start, index_x_end, index_z_start, index_z_end, index_y_start, index_y_end: ", index_x_start, index_x_end, index_z_start, index_z_end, index_y_start, index_y_end)

        print("Starting enhancement")
        # self.enhanced_layer = self.viewer.add_image(self.enhanced_zarr, name="Enhanced Image", scale=(1, 1, 1))

        QApplication.processEvents()
        patch_indices = list(range(index_x_start, index_x_end))

        # Check if GPU should be used based on args
        use_gpu = self.args.gpu if hasattr(self.args, 'gpu') else False

        # Convert latent data to torch tensor and move to appropriate device
        latent_data = self.latent[:, :, :, :, index_z_start:index_z_end, :, index_y_start:index_y_end]
        if use_gpu and torch.cuda.is_available():
            print("Using GPU for enhancement")
            latent_tensor = torch.from_numpy(latent_data).cuda()
        else:
            print("Using CPU for enhancement")
            latent_tensor = torch.from_numpy(latent_data)

        self.enhance_worker = EnhanceWorker(self.iter_enhance_by_yz_slice,
                                            latent_tensor,
                                            patch_indices,
                                            augmentation=self.args.augmentation_method,
                                            nz_start=index_z_start, nz_end=index_z_end, ny_start=index_y_start,
                                            ny_end=index_y_end, S0=S0)
        # Connect signal to update enhanced_zarr when each patch is processed
        self.enhance_worker.patch_finished.connect(self.update_enhanced_zarr)
        self.enhance_worker.finished.connect(lambda: print("Enhancement worker finished."))
        self.enhance_worker.error.connect(lambda err: print("Enhancement error:", err))
        self.enhance_worker.start()

    def register_model(self):
        self._update_params_to_processer()
        self.registered_model = False
        self.register_model_button.setEnabled(False)
        self.loader_thread = ModelLoaderThread(self.image_processer)
        self.loader_thread.finished.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()




class MainWidget(QWidget):
    def __init__(self, viewer, image_processer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()

        self.yaml_config_page = YAMLConfigPage()
        self.args_panel = ArgsPanel()
        self.visualization_page = VisualizationPage(viewer, self.yaml_config_page, image_processer)
        self.analysis_page = AnalysisPage(self.visualization_page)

        def scroll_wrap(widget):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            return scroll

        self.tab_widget.addTab(scroll_wrap(self.yaml_config_page), "YAML Config")
        self.tab_widget.addTab(scroll_wrap(self.args_panel), "Process Args")
        self.tab_widget.addTab(scroll_wrap(self.visualization_page), "Visualization")
        self.tab_widget.addTab(scroll_wrap(self.analysis_page), "Analysis")

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

        self.yaml_config_page.yaml_updated.connect(self.args_panel.update_option_choices)
        self.args_panel.args_updated.connect(self.visualization_page.update_args)
        self.yaml_config_page.yaml_updated.connect(lambda _: self.visualization_page.update_buttons_state())


if __name__ == '__main__':
    viewer = napari.Viewer()
    micro_processer = MicroTest()
    main_widget = MainWidget(viewer, micro_processer)
    viewer.window.add_dock_widget(main_widget, name="Control Panel")
    napari.run()
