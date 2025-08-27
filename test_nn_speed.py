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
else:
    # Optimize CUDA memory allocation
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Use memory pool to reduce allocation overhead
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print("Model loaded successfully!")

# Timing variables
frame_count = 0
total_nn_time = 0
start_time = time()

# Detailed timing accumulators
total_times = {
    'preprocessing': 0.0,
    'to_device': 0.0,
    'nn_inference': 0.0,
    'cuda_sync': 0.0,
    'postprocessing': 0.0,
    'to_cpu': 0.0,
    'memory_cleanup': 0.0
}

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
        global frame_count, total_nn_time, total_times
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        
        frame_count += 1
        
        # Apply NN upscaling and measure detailed timing
        nn_start = time()
        upscaled_frame, frame_times = self.apply_nn_upscaling_detailed(frame)
        nn_time = time() - nn_start
        total_nn_time += nn_time
        
        # Accumulate detailed times
        for key, value in frame_times.items():
            total_times[key] += value
        
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
            avg_inference = total_times['nn_inference'] / frame_count
            self.setWindowTitle(f"NN Speed Test - Frame {frame_count} | Total: {avg_nn_time:.3f}s | NN: {avg_inference:.3f}s | FPS: {effective_fps:.1f}")
    
    def apply_nn_upscaling_detailed(self, frame):
        global model, device
        
        times = {}
        
        # Step 1: Preprocessing (BGR to RGB, PIL conversion, tensor conversion)
        preprocess_start = time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)
        input_tensor = TF.to_tensor(pil_frame).unsqueeze(0)
        times['preprocessing'] = time() - preprocess_start
        
        # Step 2: Transfer to device
        to_device_start = time()
        input_tensor = input_tensor.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times['to_device'] = time() - to_device_start
        
        # Step 3: NN Inference
        inference_start = time()
        with torch.no_grad():
            output_tensor = model(input_tensor)
        times['nn_inference'] = time() - inference_start
        
        # Step 4: CUDA synchronization (if using GPU)
        sync_start = time()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times['cuda_sync'] = time() - sync_start
        
        # Step 5: Postprocessing (tensor operations)
        postprocess_start = time()
        output_tensor = output_tensor.squeeze(0).mul(255).clamp(0, 255).byte()
        output_tensor = output_tensor.permute(1, 2, 0)
        times['postprocessing'] = time() - postprocess_start
        
        # Step 6: Transfer to CPU
        to_cpu_start = time()
        output_frame = output_tensor.cpu().numpy()
        times['to_cpu'] = time() - to_cpu_start
        
        # Step 7: Final conversion and memory cleanup
        cleanup_start = time()
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Minimal memory cleanup - only delete references
        del input_tensor, output_tensor
        # Clear cache much less frequently to reduce sync overhead
        if device.type == 'cuda' and frame_count % 50 == 0:
            torch.cuda.empty_cache()
        times['memory_cleanup'] = time() - cleanup_start
        
        return output_frame, times
    
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
        global frame_count, total_nn_time, start_time, total_times
        
        total_time = time() - start_time
        avg_nn_time = total_nn_time / frame_count if frame_count > 0 else 0
        effective_fps = 1.0 / avg_nn_time if avg_nn_time > 0 else 0
        overall_fps = frame_count / total_time if total_time > 0 else 0
        
        # Calculate average times for each step
        avg_times = {key: total_times[key] / frame_count for key in total_times.keys()}
        
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Overall FPS: {overall_fps:.2f}")
        print(f"Average total NN time: {avg_nn_time:.3f} seconds")
        print(f"Effective NN FPS: {effective_fps:.2f}")
        print(f"NN processing overhead: {(avg_nn_time / (1/30) * 100):.1f}% of 30fps budget")
        print()
        print("DETAILED TIMING BREAKDOWN (average per frame):")
        print(f"  1. Preprocessing:       {avg_times['preprocessing']*1000:8.3f} ms")
        print(f"  2. Transfer to device:  {avg_times['to_device']*1000:8.3f} ms")
        print(f"  3. NN Inference:        {avg_times['nn_inference']*1000:8.3f} ms")
        print(f"  4. CUDA Synchronization:{avg_times['cuda_sync']*1000:8.3f} ms")
        print(f"  5. Postprocessing:      {avg_times['postprocessing']*1000:8.3f} ms")
        print(f"  6. Transfer to CPU:     {avg_times['to_cpu']*1000:8.3f} ms")
        print(f"  7. Memory cleanup:      {avg_times['memory_cleanup']*1000:8.3f} ms")
        print(f"     TOTAL:               {avg_nn_time*1000:8.3f} ms")
        print()
        print("PERFORMANCE ANALYSIS:")
        bottleneck = max(avg_times.items(), key=lambda x: x[1])
        print(f"  Bottleneck step: {bottleneck[0]} ({bottleneck[1]*1000:.3f} ms)")
        print(f"  NN inference ratio: {(avg_times['nn_inference']/avg_nn_time*100):.1f}% of total time")
        print(f"  Data transfer ratio: {((avg_times['to_device']+avg_times['to_cpu'])/avg_nn_time*100):.1f}% of total time")
        print("="*60)

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
