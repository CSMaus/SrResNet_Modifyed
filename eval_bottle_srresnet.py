import time

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from custom_srresnet import _NetX2, _NetX2Eff, _NetGS

# model_path = "model/srbottle_resnet-FT-BS4-EP30.pth"
model_path = "model/srresnet_Small-FT-BS2-EP30-Bl[0, 15].pth"
# model_path = "model/srbottle_resnet-FT-BS6-EP50-B2-Ch48.pth"  # srbottle_resnet-FT-BS4-EP20.pth"
input_image_path = "imgs/00076.png"  # step_4_vibrance.jpg"  # 00076-2.5.png"  # 00076.png
# output_path = f"imgs/{input_image_path[5:-4]}-output_bottle_1R2B48.png"
output_path = f"imgs/test21-FTsmall.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_blocks = 2  # 2 default
num_channels = 24  # 32 default

# model = _NetX2Eff(num_blocks, num_channels)
model = _NetGS(num_blocks)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
'''
if "model" in checkpoint:
    state_dict = checkpoint["model"].state_dict()
else:
    state_dict = checkpoint'''

filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

model.load_state_dict(filtered_state_dict, strict=False)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

st = time.time()
input_tensor = load_image(input_image_path).to(device)

with torch.no_grad():
    output_tensor = model(input_tensor)

output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
print("Image evaluation took: ", time.time() - st, " sec")
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"Super-resolved image saved at: {output_path}")
