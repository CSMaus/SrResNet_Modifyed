import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QLCDNumber
from PyQt6.QtWidgets import QCheckBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import pyqtSignal
import os
import torch
from custom_srresnet import _NetG
from PIL import Image
import torchvision.transforms.functional as TF

model_path = "model/srresnet_finetuned-BS2-EP30.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = _NetG().to(device)
checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
model.eval()
# model warm-up (hope it'll work if I'll run it here)
dummy_input = torch.randn(1, 3, 924, 706).to(device)
with torch.no_grad():
    model(dummy_input)



# these are parameters for enhancement low quality very bad video
exposure = 1.2
brightness = 10 # 110-100
contrast = 50  # 28 # 130 - 100
vibrance = 1.5
hue = 5
saturation = 29
lightness = 10  # 108 - 100
clip_limit = 13.5  # was 5
tile_grid_size = 71  # was 12
current_frame = 0
total_frames = 1
is_frame_reset = False
do_save_frame = False
use_nn_upscaling = False
do_nn_first = False  # do NN upscaling before CLAHE

current_directory = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.normpath(os.path.join(current_directory, "../Data/Weld_VIdeo/"))
up_ds_folder = os.path.normpath(os.path.join(current_directory, "../Data/UpVideoTest/"))

def adjust_exposure(frame):
    global exposure
    """
    Apply exposure correction using gamma correction.
    exposure > 1.0 -> brighter
    exposure < 1.0 -> darker
    """
    gamma = 1.0 / exposure  # Inverse for OpenCV gamma correction
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, lookup_table)


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_adjustments(frame):
    global brightness, contrast, vibrance, hue, saturation, lightness
    frame = apply_clahe(frame)
    img = np.int16(frame)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    xval = np.arange(0, 256)
    lut = (255 * np.tanh(vibrance * xval / 255) / np.tanh(1) + 0.5).astype(np.uint8)
    s = cv2.LUT(s, lut)

    h = (h.astype(int) + hue) % 180
    h = h.astype(np.uint8)

    s = cv2.add(s, saturation)
    v = cv2.add(v, lightness)

    adjusted_hsv = cv2.merge([h, s, v])
    adjusted_frame = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    adjusted_frame = adjust_exposure(adjusted_frame)
    return adjusted_frame


def crop_frame(frame, left_crop=80, right_crop=80, top_crop=0, bottom_crop=0):
    h, w = frame.shape[:2]
    left = int(left_crop)
    right = int(w - right_crop)
    top = int(top_crop)
    bottom = int(h - bottom_crop)
    return frame[top:bottom, left:right]


def downscale_frame(frame, scale_coef=4):
    return cv2.resize(frame, (0, 0), fx=1/scale_coef, fy=1/scale_coef, interpolation=cv2.INTER_AREA)



class VideoProcessor(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("Video Processor")
        # self.setGeometry(200, 200, )
        global total_frames

        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", total_frames)
        self.original_label = QLabel(self)
        self.processed_label = QLabel(self)

        layout = QHBoxLayout()
        layout.addWidget(self.original_label)
        layout.addWidget(self.processed_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

        self.sliders_window = SlidersWindow()
        self.sliders_window.show()

    def update_frame(self):
        global is_frame_reset, current_frame, do_save_frame, up_ds_folder, use_nn_upscaling

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        if is_frame_reset:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            # is_frame_reset = False

        if not do_nn_first:
            processed_frame = apply_adjustments(frame)
            if use_nn_upscaling:
                processed_frame = self.apply_nn_upscaling(processed_frame)

            original_image = self.convert_to_qt_image(frame)
            processed_image = self.convert_to_qt_image(processed_frame)
        else:
            if use_nn_upscaling:
                processed_frame = self.apply_nn_upscaling(frame)
                processed_frame = apply_adjustments(processed_frame)
            else:
                processed_frame = apply_adjustments(frame)

            original_image = self.convert_to_qt_image(frame)
            processed_image = self.convert_to_qt_image(processed_frame)


        self.original_label.setPixmap(original_image.scaled(
            self.original_label.width(), self.original_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
        # self.processed_label.setPixmap(processed_image)
        # to resize videos
        self.processed_label.setPixmap(processed_image.scaled(
            self.processed_label.width(), self.processed_label.height(), Qt.AspectRatioMode.KeepAspectRatio))


        if do_save_frame:
            img_name = f"{current_frame}.png"
            img_name_up = f"up_{current_frame:06d}.png"
            save_path = os.path.join(up_ds_folder, img_name)
            cv2.imwrite(save_path, frame)
            cv2.imwrite(os.path.join(up_ds_folder, img_name_up), processed_frame)
            print(f"Processed image saved at: {save_path}")
            do_save_frame = False

    def apply_nn_upscaling(self, frame):
        global model, device
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = TF.to_tensor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = model(frame)
            output_tensor = output_tensor.squeeze(0).mul(255).clamp(0, 255).byte()
            output_tensor = output_tensor.permute(1, 2, 0)
            torch.cuda.synchronize()
            # output_tensor = output_tensor.to(torch.half)
        out_frame = output_tensor.cpu().numpy()
        # out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        return out_frame

    def convert_to_qt_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image)

    def keyPressEvent(self, event):
        global do_save_frame
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
        elif event.key() == Qt.Key.Key_S:
            # save_path = os.path.join(os.path.dirname(self.image_path), "processed_" + os.path.basename(self.image_path))
            # cv2.imwrite(save_path, self.processed_frame)
            # print(f"Processed image saved at: {save_path}")
            do_save_frame = True
            print("TODO: define the path and name for saving frames")
        elif event.key() == Qt.Key.Key_P:
            print("\nCurrent Parameters:")
            print("Exposure: ", exposure)
            print("Brightness: ", brightness)
            print("Contrast: ", contrast)
            print("Vibrance: ", vibrance)
            print("Hue: ", hue)
            print("Saturation: ", saturation)
            print("Lightness: ", lightness)
            print("Clip Limit: ", clip_limit)
            print("Tile Grid Size: ", tile_grid_size)

class SlidersWindow(QWidget):
    frame_reset_signal = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sliders")
        self.setGeometry(100, 100, 600, 400)  # (position at 100 x100, width 400, height 600)

        self.exposure_slider = self.create_slider("Exposure", 10, 30, exposure, self.update_exposure)
        self.brightness_slider = self.create_slider("Brightness", 0, 200, brightness, self.update_brightness)
        self.contrast_slider = self.create_slider("Contrast", 0, 200, contrast, self.update_contrast)
        self.vibrance_slider = self.create_slider("Vibrance", 0, 30, int(vibrance * 10), self.update_vibrance)
        self.hue_slider = self.create_slider("Hue", 0, 100, hue, self.update_hue)
        self.saturation_slider = self.create_slider("Saturation", 0, 100, saturation, self.update_saturation)
        self.lightness_slider = self.create_slider("Lightness", 0, 200, lightness, self.update_lightness)
        self.clip_limit_slider = self.create_slider("CLAHE Clip Limit", 1, 150, int(clip_limit * 10), self.update_clip_limit)
        self.tile_grid_slider = self.create_slider("CLAHE Tile Grid Size", 1, 100, tile_grid_size, self.update_tile_grid)
        self.frame_reset_slider = self.create_slider("Frame Reset", 0, total_frames-1, 0, self.update_frame_reset)

        # checkbox
        self.checkboxUpFirstNN = QCheckBox("Do Upscaling First")
        self.checkboxUpFirstNN.setChecked(False)
        self.checkboxUpFirstNN.stateChanged.connect(self.update_do_nn_first)
        # checkbox
        self.checkboxNN = QCheckBox("Apply NN upscaling")
        self.checkboxNN.setChecked(False)
        self.checkboxNN.stateChanged.connect(self.update_use_nn_upscaling)
        # checkbox
        self.checkbox = QCheckBox("Lock Frame to Slider")
        self.checkbox.setChecked(False)
        self.checkbox.stateChanged.connect(self.update_checkbox)

        layout = QVBoxLayout()
        layout.addWidget(self.checkboxUpFirstNN)
        layout.addWidget(self.exposure_slider)
        layout.addWidget(self.brightness_slider)
        layout.addWidget(self.contrast_slider)
        layout.addWidget(self.vibrance_slider)
        layout.addWidget(self.hue_slider)
        layout.addWidget(self.saturation_slider)
        layout.addWidget(self.lightness_slider)
        layout.addWidget(self.clip_limit_slider)
        layout.addWidget(self.tile_grid_slider)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.frame_reset_slider)
        layout.addWidget(self.checkboxNN)

        self.setLayout(layout)

    def create_slider(self, name, min_value, max_value, default_value, callback):
        slider_container = QWidget()
        layout = QHBoxLayout()

        label = QLabel(name)
        lcd = QLCDNumber()
        lcd.display(default_value)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.valueChanged.connect(lambda value: lcd.display(value))
        slider.valueChanged.connect(callback)

        layout.addWidget(label)
        layout.addWidget(lcd)
        layout.addWidget(slider)
        slider_container.setLayout(layout)
        return slider_container

    def update_exposure(self, value):
        global exposure
        exposure = value / 10

    def update_brightness(self, value):
        global brightness
        brightness = value - 100

    def update_contrast(self, value):
        global contrast
        contrast = value - 100

    def update_vibrance(self, value):
        global vibrance
        vibrance = value / 10

    def update_hue(self, value):
        global hue
        hue = value

    def update_saturation(self, value):
        global saturation
        saturation = value - 100

    def update_lightness(self, value):
        global lightness
        lightness = value - 100

    def update_clip_limit(self, value):
        global clip_limit
        clip_limit = max(1, value) / 10

    def update_tile_grid(self, value):
        global tile_grid_size
        tile_grid_size = max(1, value)

    def update_frame_reset(self, value):
        global current_frame
        current_frame = max(1, value)

    def update_checkbox(self, state):
        global is_frame_reset
        is_frame_reset = state

    def update_use_nn_upscaling(self, state):
        global use_nn_upscaling
        use_nn_upscaling = state

    def update_do_nn_first(self, state):
        global do_nn_first
        do_nn_first = state
        # is_frame_reset = state == Qt.CheckState.Checked

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_name = "Weld_Video_2023-04-20_01-55-23_Camera01.avi.avi"
    # video_name = "low_quality.mp4"
    # video_name = "HighQuality.mp4"
    video_path = os.path.join(datapath, video_name)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()


    main_window = VideoProcessor(video_path)
    main_window.show()

    sys.exit(app.exec())




