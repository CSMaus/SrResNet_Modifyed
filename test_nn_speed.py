import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import os
import torch
from custom_srresnet import _NetG
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
from time import time

# Default paths - matching original script
current_directory = os.path.dirname(os.path.abspath(__file__))
def_video_path = os.path.join(current_directory, "vid_426_240_18fps.mp4")
model_path = "model/srresnet_finetuned-BS2-EP30.pth"

parser = argparse.ArgumentParser(description="Simple NN Speed Test - Press Esc to exit and see timing stats")
parser.add_argument("--video_path", type=str, default=def_video_path, help="Path to the video file")
parser.add_argument("--model_path", type=str, default=model_path, help="Path to the model file")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading model...")
model = _NetG().to(device)
checkpoint = torch.load(args.model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
model.eval()

# Memory optimization for CPU
if device.type == 'cpu':
    torch.set_num_threads(2)

print("Model loaded successfully!")

# Timing variables
frame_count = 0
total_nn_time = 0
start_time = time()

class SimpleVideoProcessor(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.setWindowTitle("NN Speed Test - Original vs Upscaled")
        self.setGeometry(100, 100, 1200, 600)
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            sys.exit(1)
            
        # Get video info
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {self.total_frames} frames at {self.fps} FPS")
        
        # Create UI
        self.original_label = QLabel("Original")
        self.processed_label = QLabel("NN Upscaled")
        
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.processed_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid black")
        self.processed_label.setStyleSheet("border: 1px solid black")
        
        layout = QHBoxLayout()
        layout.addWidget(self.original_label)
        layout.addWidget(self.processed_label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        
    def update_frame(self):
        global frame_count, total_nn_time
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        
        frame_count += 1
        
        # Apply NN upscaling and measure time
        nn_start = time()
        upscaled_frame = self.apply_nn_upscaling(frame)
        nn_time = time() - nn_start
        total_nn_time += nn_time
        
        # Convert to Qt images
        original_image = self.convert_to_qt_image(frame)
        processed_image = self.convert_to_qt_image(upscaled_frame)
        
        # Display images (scaled to fit)
        self.original_label.setPixmap(original_image.scaled(
            self.original_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.processed_label.setPixmap(processed_image.scaled(
            self.processed_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        # Update window title with current stats
        if frame_count % 10 == 0:  # Update every 10 frames
            avg_nn_time = total_nn_time / frame_count
            effective_fps = 1.0 / avg_nn_time if avg_nn_time > 0 else 0
            self.setWindowTitle(f"NN Speed Test - Frame {frame_count} | Avg NN Time: {avg_nn_time:.3f}s | Effective FPS: {effective_fps:.1f}")
    
    def apply_nn_upscaling(self, frame):
        global model, device
        
        # Convert frame to tensor
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = TF.to_tensor(pil_frame).unsqueeze(0).to(device)
        
        # Apply model
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.squeeze(0).mul(255).clamp(0, 255).byte()
            output_tensor = output_tensor.permute(1, 2, 0)
        
        # Convert back to numpy array
        output_frame = output_tensor.cpu().numpy()
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Clean up memory
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return output_frame
    
    def convert_to_qt_image(self, cv_image):
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.print_stats()
            QApplication.quit()
    
    def print_stats(self):
        global frame_count, total_nn_time, start_time
        
        total_time = time() - start_time
        avg_nn_time = total_nn_time / frame_count if frame_count > 0 else 0
        effective_fps = 1.0 / avg_nn_time if avg_nn_time > 0 else 0
        overall_fps = frame_count / total_time if total_time > 0 else 0
        
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Overall FPS: {overall_fps:.2f}")
        print(f"Average NN processing time: {avg_nn_time:.3f} seconds")
        print(f"Effective NN FPS: {effective_fps:.2f}")
        print(f"NN processing overhead: {(avg_nn_time / (1/30) * 100):.1f}% of 30fps budget")
        print("="*50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        print("Available files in current directory:")
        for f in os.listdir(current_directory):
            if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"  {f}")
        sys.exit(1)
    
    main_window = SimpleVideoProcessor(args.video_path)
    main_window.show()
    
    print("Press ESC to exit and see detailed performance statistics")
    sys.exit(app.exec())
