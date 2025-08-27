import sys
import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import os
import torch
from custom_srresnet import _NetG
from PIL import Image
import torchvision.transforms.functional as TF
import argparse
from time import time

# Default paths
current_directory = os.path.dirname(os.path.abspath(__file__))
def_video_path = os.path.join(current_directory, "vid_426_240_18fps.mp4")
model_path = "model/srresnet_finetuned-BS2-EP30.pth"

parser = argparse.ArgumentParser(description="GPU-Direct NN Speed Test - Press ESC to exit")
parser.add_argument("--video_path", type=str, default=def_video_path, help="Path to the video file")
parser.add_argument("--model_path", type=str, default=model_path, help="Path to the model file")
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")

# Load model
print("Loading model...")
model = _NetG().to(device)
checkpoint = torch.load(args.model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)
model.eval()

# Memory optimization
if device.type == 'cpu':
    torch.set_num_threads(2)
else:
    torch.backends.cudnn.benchmark = True

print("Model loaded successfully!")

# Timing variables
frame_count = 0
total_times = {
    'video_read': 0.0,
    'preprocessing': 0.0,
    'to_device': 0.0,
    'nn_inference': 0.0,
    'cuda_sync': 0.0,
    'gpu_to_opengl': 0.0,
    'opengl_render': 0.0,
    'total_frame': 0.0
}
start_time = time()

# OpenGL shaders for texture rendering
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D ourTexture;

void main()
{
    FragColor = texture(ourTexture, TexCoord);
}
"""

class GPUDirectVideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            sys.exit(1)
            
        # Get video info
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {self.width}x{self.height}, {self.total_frames} frames at {self.fps} FPS")
        
        # Initialize pygame and OpenGL
        self.init_opengl()
        
        # Setup OpenGL textures and shaders
        self.setup_opengl_rendering()
        
        # Pre-allocate CUDA tensors to avoid repeated allocation
        self.input_tensor_template = torch.zeros(1, 3, self.height, self.width, device=device, dtype=torch.float32)
        
    def init_opengl(self):
        pygame.init()
        
        # Calculate window size (side by side display)
        window_width = self.width * 2 + 50  # Space for both images plus gap
        window_height = max(self.height, 600)
        
        self.screen = pygame.display.set_mode((window_width, window_height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("GPU-Direct NN Processing")
        
        # OpenGL setup
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, window_width, window_height)
        
    def setup_opengl_rendering(self):
        # Compile shaders
        self.shader_program = compileProgram(
            compileShader(vertex_shader, GL_VERTEX_SHADER),
            compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        )
        
        # Create textures for original and processed frames
        self.original_texture = glGenTextures(1)
        self.processed_texture = glGenTextures(1)
        
        # Setup texture parameters
        for texture in [self.original_texture, self.processed_texture]:
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Create vertex data for two quads (left and right)
        # Left quad (original)
        left_vertices = np.array([
            # positions        # texture coords
            -1.0, -1.0, 0.0,   0.0, 0.0,  # bottom left
             0.0, -1.0, 0.0,   1.0, 0.0,  # bottom right
             0.0,  1.0, 0.0,   1.0, 1.0,  # top right
            -1.0,  1.0, 0.0,   0.0, 1.0   # top left
        ], dtype=np.float32)
        
        # Right quad (processed)
        right_vertices = np.array([
            # positions        # texture coords
             0.0, -1.0, 0.0,   0.0, 0.0,  # bottom left
             1.0, -1.0, 0.0,   1.0, 0.0,  # bottom right
             1.0,  1.0, 0.0,   1.0, 1.0,  # top right
             0.0,  1.0, 0.0,   0.0, 1.0   # top left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Setup VAOs and VBOs for both quads
        self.setup_quad_buffers(left_vertices, indices, 'left')
        self.setup_quad_buffers(right_vertices, indices, 'right')
        
    def setup_quad_buffers(self, vertices, indices, side):
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        
        glBindVertexArray(vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        
        if side == 'left':
            self.left_vao = vao
        else:
            self.right_vao = vao
            
    def process_frame(self):
        global frame_count, total_times
        
        frame_start = time()
        
        # Step 1: Read video frame
        read_start = time()
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return False
        total_times['video_read'] += time() - read_start
        
        frame_count += 1
        
        # Step 2: Preprocessing (minimize CPU work)
        preprocess_start = time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor efficiently
        frame_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        input_tensor = frame_tensor.unsqueeze(0)
        total_times['preprocessing'] += time() - preprocess_start
        
        # Step 3: Transfer to device
        to_device_start = time()
        input_tensor = input_tensor.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_times['to_device'] += time() - to_device_start
        
        # Step 4: NN Inference
        inference_start = time()
        with torch.no_grad():
            output_tensor = model(input_tensor)
        total_times['nn_inference'] += time() - inference_start
        
        # Step 5: CUDA synchronization
        sync_start = time()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_times['cuda_sync'] += time() - sync_start
        
        # Step 6: GPU to OpenGL (NO CPU TRANSFER!)
        gpu_to_gl_start = time()
        
        # Process output tensor on GPU
        output_tensor = output_tensor.squeeze(0).clamp(0, 1)
        output_tensor = (output_tensor * 255).byte()
        
        # Convert tensors to numpy for OpenGL (this is the only CPU transfer, but much smaller)
        original_np = (frame_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy()
        processed_np = output_tensor.permute(1, 2, 0).cpu().numpy()
        
        total_times['gpu_to_opengl'] += time() - gpu_to_gl_start
        
        # Step 7: OpenGL rendering
        render_start = time()
        self.render_frame(original_np, processed_np)
        total_times['opengl_render'] += time() - render_start
        
        # Cleanup
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        total_times['total_frame'] += time() - frame_start
        
        return True
        
    def render_frame(self, original_frame, processed_frame):
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(self.shader_program)
        
        # Render original frame (left side)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.original_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, original_frame)
        
        glBindVertexArray(self.left_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # Render processed frame (right side)
        glBindTexture(GL_TEXTURE_2D, self.processed_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, processed_frame.shape[1], processed_frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, processed_frame)
        
        glBindVertexArray(self.right_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        pygame.display.flip()
        
    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        print("Running GPU-direct processing. Press ESC to exit and see stats.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            if not self.process_frame():
                break
                
            # Update title with performance info
            if frame_count % 30 == 0:
                avg_total = total_times['total_frame'] / frame_count
                avg_nn = total_times['nn_inference'] / frame_count
                fps = 1.0 / avg_total if avg_total > 0 else 0
                pygame.display.set_caption(f"GPU-Direct Processing - Frame {frame_count} | Total: {avg_total:.3f}s | NN: {avg_nn:.3f}s | FPS: {fps:.1f}")
            
            # Target 30 FPS
            clock.tick(30)
        
        self.print_stats()
        pygame.quit()
        
    def print_stats(self):
        global frame_count, total_times, start_time
        
        total_runtime = time() - start_time
        avg_times = {key: total_times[key] / frame_count for key in total_times.keys()}
        
        print("\n" + "="*60)
        print("GPU-DIRECT PERFORMANCE STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total runtime: {total_runtime:.2f} seconds")
        print(f"Overall FPS: {frame_count / total_runtime:.2f}")
        print(f"Average frame time: {avg_times['total_frame']:.3f} seconds")
        print(f"Effective FPS: {1.0 / avg_times['total_frame']:.2f}")
        print()
        print("DETAILED TIMING BREAKDOWN (average per frame):")
        print(f"  1. Video read:          {avg_times['video_read']*1000:8.3f} ms")
        print(f"  2. Preprocessing:       {avg_times['preprocessing']*1000:8.3f} ms")
        print(f"  3. Transfer to device:  {avg_times['to_device']*1000:8.3f} ms")
        print(f"  4. NN Inference:        {avg_times['nn_inference']*1000:8.3f} ms")
        print(f"  5. CUDA Synchronization:{avg_times['cuda_sync']*1000:8.3f} ms")
        print(f"  6. GPU to OpenGL:       {avg_times['gpu_to_opengl']*1000:8.3f} ms")
        print(f"  7. OpenGL rendering:    {avg_times['opengl_render']*1000:8.3f} ms")
        print(f"     TOTAL:               {avg_times['total_frame']*1000:8.3f} ms")
        print()
        bottleneck = max(avg_times.items(), key=lambda x: x[1])
        print(f"Bottleneck step: {bottleneck[0]} ({bottleneck[1]*1000:.3f} ms)")
        print(f"NN inference ratio: {(avg_times['nn_inference']/avg_times['total_frame']*100):.1f}% of total time")
        print("="*60)

if __name__ == "__main__":
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        processor = GPUDirectVideoProcessor(args.video_path)
        processor.run()
    except Exception as e:
        print(f"Error: {e}")
        pygame.quit()
        sys.exit(1)
