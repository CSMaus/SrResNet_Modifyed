import time

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from srresnet import _NetG
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

# model_path = "model/model_srresnet.pth"
# model_path = "model/srresnet_finetuned-BS2-EP30.pth"  # srresnet_finetuned.pth"
# model_path = "model/srresnet-LL2NL3.0_finetuned-BS2-EP50.pth"  # srresnet_finetuned.pth"

model_path = "model/srresnet_finetuned-BS2-EP30.pth"
input_image_path = "imgs/00075.png"  # step_4_vibrance.jpg"  # 00076-2.5.png"  # 00076.png
output_path = f"imgs/{input_image_path[5:-4]}_output-FT-BS2-EP30-SPEEDUP.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = _NetG()

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
print("Input image loaded in: ", time.time() - st, " sec")

with torch.no_grad():
    testt = time.time()
    model(input_tensor)
    print("Model warm-up took: ", time.time() - testt, " sec")
    st = time.time()
    output_tensor = model(input_tensor)
    print("Model inference took: ", time.time() - st, " sec")



# image_conversion_back_time = time.time()

# output_image = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
# output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
# trying to optimize:
# output_image = TF.to_pil_image(output_tensor.squeeze(0).cpu())  # also slow
# output_image.save(output_path)

# output_tensor = output_tensor.squeeze(0).mul(255).clamp(0, 255).byte()  # Convert on GPU first
# output_image_perm = output_tensor.permute(1, 2, 0)
# output_tensor = output_tensor.squeeze(0).detach().mul(255).clamp(0, 255).to(torch.uint8)
output_tensor = output_tensor.squeeze(0).detach().mul(255).clamp(0, 255).to(torch.uint8)
output_image = output_tensor.permute(1, 2, 0) # .numpy()

# print("Image squeeze and permute took: ", time.time() - image_conversion_back_time, " sec")


import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

def show_image_with_matplotlib(output_tensor):
    output_array = cuda.from_device_like(output_tensor, np.empty_like(output_tensor))

    plt.imshow(output_array)
    plt.axis("off")
    plt.show()

topil = time.time()
show_image_with_matplotlib(output_image)
# output_image = output_image_perm.cpu().numpy()  # Move to CPU after conversion

# still too much time
# output_image_perm = output_tensor.permute(1, 2, 0).contiguous()
# output_image = output_image_perm.to("cpu", non_blocking=True).numpy()
# output_image = to_pil_image(output_tensor)


print("Image conversion to PIL took: ", time.time() - topil, " sec")
# print("Image conversion back took: ", time.time() - image_conversion_back_time, " sec")
cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
print(f"Super-resolved image saved at: {output_path}")
