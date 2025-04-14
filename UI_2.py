import os
import sys
import zarr
import argparse
from io import StringIO
import napari
import torch
import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QPushButton, QFileDialog,
    QHBoxLayout, QPlainTextEdit, QFormLayout, QLineEdit, QCheckBox, QComboBox, QApplication
)
from qtpy.QtCore import Qt, Signal

from test_assemble import MicroTest
from utils.data_utils import _CHECK_PARAMS, DataNormalization
import tifffile as tiff
from qtpy.QtCore import QThread, Signal

# Using other thread to register model, not effect main thread
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

# keep yaml same format
yaml_ruamel = YAML()
yaml_ruamel.preserve_quotes = True

class ArgsPanel(QWidget):
    # 與 YAMLConfigPage 類似，定義一個 signal
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
        self.parser.add_argument('--save', nargs='+', choices=['ori', 'recon', 'xy'], required=False,
                                 help="assign image to save: --save ori recon")
        self.parser.add_argument('--image_datatype', type=str, choices=["float32", "float16", "uint16","uint8"],
                                 default="float32")
        self.parser.add_argument('--augmentation', type=str, default="decode")
        self.parser.add_argument('--reslice', action='store_true', default=False)
        self.parser.add_argument('--host', type=str, default='dummy')
        self.parser.add_argument('--port', type=str, default='dummy')
        self.parser.add_argument('--assemble_method', type=str, default='tiff',
                                 help='tiff or zarr method while assemble images')
        self.parser.add_argument('--roi', type=str, default='')
        self.parser.add_argument('--targets', nargs='+', default=None, required=False, help="assign target to assemble")

        self.args = self.parser.parse_args([])

        layout = QFormLayout()
        self.widgets = {}

        for action in self.parser._actions:
            if action.dest == 'help':
                continue
            # 若為 option 參數則使用 QComboBox，之後由 YAML 更新選項
            if action.dest == "option":
                widget = QComboBox()
            # 如果有 choices 且非多值，則用 QComboBox
            elif action.choices is not None and action.nargs is None:
                widget = QComboBox()
                for choice in action.choices:
                    widget.addItem(str(choice))
            elif isinstance(action.default, bool):
                widget = QCheckBox()
                widget.setChecked(action.default)
            elif action.nargs in ['+', '*']:
                default_text = ", ".join(action.default) if action.default else ""
                widget = QLineEdit(default_text)
            else:
                widget = QLineEdit(str(action.default))
            layout.addRow(action.dest, widget)
            self.widgets[action.dest] = widget

        self.update_button = QPushButton("Update Args")
        self.update_button.clicked.connect(self.on_update)
        v_layout = QVBoxLayout()
        v_layout.addLayout(layout)
        v_layout.addWidget(self.update_button)
        self.setLayout(v_layout)

    def update_option_choices(self, yaml_config):
        """
        根據 YAML config 更新 'option' 的選項，
        YAML config 中除了 'DEFAULT' 外的 key 就是可用選項
        """
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
        # 根據 widget 的值更新 args 物件
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
        print("Updated args:", self.args)
        self.args_updated.emit(self.args)

class YAMLConfigPage(QWidget):
    # 新增 signal，當 YAML config 更新時發出，傳遞字典資料
    yaml_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.yaml = yaml_ruamel  # 使用 ruamel.yaml 的 instance
        layout = QVBoxLayout()
        # 文字編輯器用來顯示 YAML 內容
        self.editor = QPlainTextEdit()
        layout.addWidget(self.editor)

        # Load YAML 與 Update YAML 按鈕
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load YAML")
        self.load_button.clicked.connect(self.load_yaml)
        btn_layout.addWidget(self.load_button)

        self.update_button = QPushButton("Update YAML")
        self.update_button.clicked.connect(self.update_yaml)
        btn_layout.addWidget(self.update_button)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        # 儲存 YAML 解析後的設定
        self.yaml_config = {}

    def load_yaml(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load YAML File", "", "YAML Files (*.yaml *.yml)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            self.editor.setPlainText(content)
            # 使用 ruamel.yaml 解析並保留原始格式資訊
            self.yaml_config = self.yaml.load(content)
            print(f"YAML file loaded: {filename}")
            # 發出 signal 更新 YAML config
            self.yaml_updated.emit(self.yaml_config)

    def update_yaml(self):
        content = self.editor.toPlainText()
        try:
            self.yaml_config = self.yaml.load(content)
            # print("YAML configuration updated:", self.yaml_config)
            # 發出 signal 更新 YAML config
            self.yaml_updated.emit(self.yaml_config)
        except Exception as e:
            print("Error updating YAML:", e)

class VisualizationPage(QWidget):
    def __init__(self, viewer, yaml_page, image_processer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.yaml_page = yaml_page
        self.image_processer = image_processer
        self.args = {}

        self.z_start = None
        self.z_end = None

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

        #################### Params in this class used #############
        self.registered_model = False
        self.latent_loaded = False

        self.low_sr_image = None # 那張經過upsample拿來前端顯示的圖
        # self.low_sr_image = None # 未經upsample的，拿來給模型的 # 暫時不用，有latent

        #################### Params for micro class ################


    def eventFilter(self, obj, event):
        from qtpy.QtCore import QEvent
        from qtpy.QtWidgets import QToolTip

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

    def update_args(self, args):
        """接收 ArgsPanel 更新後的參數，並檢查按鈕狀態"""
        self.args = args
        print("VisualizationPage updated with args:", self.args)
        self.update_buttons_state()

    def _update_params_to_processer(self):
        self.image_processer.args = self.args
        self.image_processer.kwargs = self.image_processer.process_config(self.yaml_page.yaml_config,
                                                                          self.image_processer.args.option)
        # This is dummy
        self.image_processer.save_image_datatype = self.image_processer.args.image_datatype
        if self.image_processer.normalization is None:
            self.image_processer.normalization = DataNormalization(
                backward_type=self.image_processer.save_image_datatype)

    # def __load_image(self):
    #     # click to choose
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "Open 3D Image", "", "Image Files (*.tif *.tiff)")
    #     if filename:
    #         self._update_params_to_processer()
    #         image = tiff.imread(filename)
    #         image = self.image_processer.normalization.forward_normalization(
    #             image, self.image_processer.kwargs["norm_method"][0], self.image_processer.kwargs['trd'][0])
    #
    #         up = torch.nn.Upsample(scale_factor=(self.image_processer.kwargs['N_resolution'], 1), mode='bilinear', align_corners=True)
    #
    #     #     for x_slice in range(image.shape[3]):
    #     #         up_image = up(image[:, :, :, x_slice, :])
    #     # self.viewer.add_image(image, name="3D Image")
    #     # print(f"Loaded image!")
    #     # self.update_buttons_state()
    #     #     print("self.image_processer.kwargs['N_resolution']: ", self.image_processer.kwargs['N_resolution'])
    #     #     print("image.shape[3]: ", image.shape[3])
    #
    #         # dynamic load image not stuck in main thread
    #         processed_slices = []
    #         layer = None
    #
    #         for x_slice in range(image.shape[3]):
    #             slice_data = image[:, :, :, x_slice, :]
    #             up_tensor = up(slice_data)
    #             up_image = np.squeeze(up_tensor.numpy(), 0)
    #             processed_slices.append(up_image)
    #
    #             current_volume = np.concatenate(processed_slices, axis=0)
    #
    #             if layer is None:
    #                 layer = self.viewer.add_image(current_volume, name="3D Image")
    #             else:
    #                 layer.data = current_volume
    #
    #             QApplication.processEvents()
    #         self.upsampled_low_sr_image = current_volume
    #         print("Loaded image!")
    #         self.update_buttons_state()

    # def load_image(self):
    #     # click to choose
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "Open 3D Image", "", "Image Files (*.tif *.tiff)")
    #     if filename:
    #         self._update_params_to_processer()
    #         image = tiff.imread(filename)
    #         image = self.image_processer.normalization.forward_normalization(
    #             image, self.image_processer.kwargs["norm_method"][0], self.image_processer.kwargs['trd'][0])
    #         zarr_image = zarr.array(np.transpose(image.numpy()[0,0,::], (1, 0, 2)), chunks=(32, 256, 256))
    #         self.viewer.add_image(zarr_image, name="3D Image", scale=(1, self.image_processer.kwargs["N_resolution"], 1))
    #
    #         print("Loaded image!")
    #         self.update_buttons_state()

    # def load_image_to_zarr_used(self):
    #     import os
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "Open 3D Image", "", "Image Files (*.tif *.tiff)")
    #     if filename:
    #         self._update_params_to_processer()
    #         # 讀取 TIFF 並正規化
    #         image = tiff.imread(filename)
    #         image = self.image_processer.normalization.forward_normalization(
    #             image, self.image_processer.kwargs["norm_method"][0],
    #             self.image_processer.kwargs["trd"][0]
    #         )
    #         # 轉置：這裡假設你需要調整軸順序
    #         transposed = np.transpose(image.numpy()[0, 0, ::], (1, 0, 2))
    #
    #         # 定義一個暫存目錄（可自行選擇合適的位置）
    #         store_path = os.path.join(os.getcwd(), "temp_image.zarr")
    #         # 如果 store_path 存在就先刪除（避免 overwrite 警告）
    #         if os.path.exists(store_path):
    #             import shutil
    #             shutil.rmtree(store_path)
    #
    #         # 寫入 zarr store，指定 chunk 大小
    #         zarr_array = zarr.open(store_path, mode='w',
    #                                shape=transposed.shape,
    #                                chunks=(32, 256, 256),
    #                                dtype=transposed.dtype)
    #         zarr_array[:] = transposed
    #
    #         # 以唯讀模式打開 zarr store（lazy loading）
    #         lazy_zarr = zarr.open(store_path, mode='r')
    #
    #         # 加入到 napari，利用 scale 參數進行視覺上的重採樣
    #         self.viewer.add_image(lazy_zarr, name="3D Image",
    #                               scale=(1, self.image_processer.kwargs["N_resolution"], 1))
    #
    #         print("Loaded image!")
    #         self.update_buttons_state()

    def load_image(self):
        self._update_params_to_processer()

        folder = QFileDialog.getExistingDirectory(self, "Select Zarr Folder", os.getcwd())
        if folder:
            lazy_zarr = zarr.open(folder, mode='r')
            self.viewer.add_image(lazy_zarr, name="3D Image",
                                  scale=(1, self.image_processer.kwargs["N_resolution"], 1))
            print("Loaded image from existing zarr store!")
            self.update_buttons_state()
        else:
            print("No folder selected!")

    def load_latent(self):
        self._update_params_to_processer()
        latent = self.image_processer.get_data()
        print(f"Loaded latent!")
        self.latent_loaded = True
        self.update_buttons_state()

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
                print("assemble_params 資料不完整！")
                return

            boundary_color = assemble_params.get("boundary_color", "red")
            boundary_line_width = assemble_params.get("boundary_line_width", 2)
            face_color = "transparent"

            if self.viewer.dims.ndisplay == 2:
                self.z_start = xrange_val[0]
                self.z_end = xrange_val[1] + dx_shape[1]

                x0 = zrange_val[0]
                x1 = zrange_val[1] + dx_shape[1]
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
                x0 = zrange_val[0]
                x1 = zrange_val[1] + dx_shape[1]
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
        self.confirm_manual_button.setVisible(True)

    def confirm_manual_boundary(self):
        if "Manual XY Boundary" not in self.viewer.layers:
            print("尚未繪製 XY 邊界。")
            return
        xy_layer = self.viewer.layers["Manual XY Boundary"]
        if not xy_layer.data or len(xy_layer.data) == 0:
            print("未找到 XY 邊界形狀。")
            return
        xy_shape = np.array(xy_layer.data[0])
        x_coords = xy_shape[:, 0]
        y_coords = xy_shape[:, 1]
        x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
        y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
        if self.z_start is None or self.z_end is None:
            print("請先確認 Z 軸的起始與截止位置。")
            return
        config = self.yaml_page.yaml_config
        default = config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        dx_shape = assemble_params.get("dx_shape", None)
        if dx_shape is None:
            print("YAML 中缺少 dx_shape，無法更新 boundary 參數。")
            return
        new_xrange = [self.z_start, self.z_end - dx_shape[0]]
        new_zrange = [x_min, x_max - dx_shape[1]]
        new_yrange = [y_min, y_max - dx_shape[2]]
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

        print("手動 boundary 更新 YAML：")
        print("xrange:", new_xrange, "zrange:", new_zrange, "yrange:", new_yrange)
        if "Manual XY Boundary" in self.viewer.layers:
            self.viewer.layers.remove("Manual XY Boundary")
        self.show_boundary()

    def show_binary_boundary(self):
        vol = np.random.rand(100, 200, 200)
        binary_volume = (vol > 0.95).astype(np.uint8)
        indices = np.array(np.nonzero(binary_volume))
        if indices.size == 0:
            print("Binary volume 中無非零元素")
            return
        min_coords = indices.min(axis=1).tolist()
        max_coords = indices.max(axis=1).tolist()
        v0 = [min_coords[0], min_coords[1], min_coords[2]]
        v1 = [min_coords[0], min_coords[1], max_coords[2]]
        v2 = [min_coords[0], max_coords[1], max_coords[2]]
        v3 = [min_coords[0], max_coords[1], min_coords[2]]
        v4 = [max_coords[0], min_coords[1], min_coords[2]]
        v5 = [max_coords[0], min_coords[1], max_coords[2]]
        v6 = [max_coords[0], max_coords[1], max_coords[2]]
        v7 = [max_coords[0], max_coords[1], min_coords[2]]
        face_front  = [v0, v1, v2, v3]
        face_back   = [v4, v5, v6, v7]
        face_left   = [v0, v1, v5, v4]
        face_right  = [v3, v2, v6, v7]
        face_top    = [v1, v2, v6, v5]
        face_bottom = [v0, v3, v7, v4]
        shape_data = [np.array(face_front), np.array(face_back),
                      np.array(face_left), np.array(face_right),
                      np.array(face_top), np.array(face_bottom)]
        shape_type = "polygon"
        face_colors = ['rgba(255,0,0,0.3)', 'rgba(0,0,255,0.3)',
                       'rgba(0,255,0,0.3)', 'rgba(255,255,0,0.3)',
                       'rgba(0,255,255,0.3)', 'rgba(255,0,255,0.3)']
        edge_colors = ['black'] * 6

        if "Binary Boundary" in self.viewer.layers:
            layer = self.viewer.layers["Binary Boundary"]
            layer.data = shape_data
            layer.face_color = face_colors
            layer.edge_color = edge_colors
            layer.edge_width = 2
            layer.shape_type = shape_type
        else:
            self.viewer.add_shapes(shape_data, shape_type=shape_type, name="Binary Boundary",
                                     face_color=face_colors, edge_color=edge_colors, edge_width=2)
        print("Binary Boundary displayed.")

    def enhance_image(self):
        default = self.yaml_page.yaml_config.get("DEFAULT", {})
        assemble_params = default.get("assemble_params", {})
        xrange_val = assemble_params.get("xrange", None)
        zrange_val = assemble_params.get("zrange", None)
        yrange_val = assemble_params.get("yrange", None)
        print("Enhance Image - Current Parameters:")
        print("xrange:", xrange_val)
        print("zrange:", zrange_val)
        print("yrange:", yrange_val)

    def register_model(self):
        self._update_params_to_processer()
        self.registered_model = False  # 尚未完成
        self.register_model_button.setEnabled(False)  # 禁用按鈕，防止重複點擊
        self.loader_thread = ModelLoaderThread(self.image_processer)
        self.loader_thread.finished.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()

    def on_model_loaded(self, model):
        self.registered_model = True
        self.register_model_button.setEnabled(True)
        self.update_buttons_state()
        print("Model loaded successfully.")

    def on_model_error(self, error_msg):
        self.register_model_button.setEnabled(True)
        print("Error loading model:", error_msg)



class MainWidget(QWidget):
    def __init__(self, viewer, image_processer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        main_layout = QVBoxLayout()
        self.tab_widget = QTabWidget()
        self.yaml_config_page = YAMLConfigPage()
        self.visualization_page = VisualizationPage(viewer, self.yaml_config_page, image_processer)
        self.args_panel = ArgsPanel()

        self.tab_widget.addTab(self.yaml_config_page, "YAML Config")
        self.tab_widget.addTab(self.args_panel, "Process Args")
        self.tab_widget.addTab(self.visualization_page, "Visualization")

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
