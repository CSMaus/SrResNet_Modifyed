import time

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from custom_srresnet import _NetX2

# model_path = "model/model_srresnet.pth"
# model_path = "model/srresnet_finetuned-BS2-EP30.pth"  # srresnet_finetuned.pth"
model_path = "model/srbottle_resnet.pth"  # srresnet_finetuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = _NetX2()

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
input_image_path = "imgs/00075.png"  # step_4_vibrance.jpg"  # 00076-2.5.png"  # 00076.png
input_tensor = load_image(input_image_path).to(device)

with torch.no_grad():
    output_tensor = model(input_tensor)

output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
print("Image evaluation took: ", time.time() - st, " sec")
output_path = f"imgs/output_bottleSR_{input_image_path[5:]}"
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"Super-resolved image saved at: {output_path}")
