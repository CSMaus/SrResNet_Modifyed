import os
import cv2
import numpy as np
from tqdm import tqdm


# params
brightness = 34
contrast = 23
vibrance = 2.6
hue = 5
saturation = 0
lightness = 20
clip_limit = 6.5
tile_grid_size = 1


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_adjustments(frame):
    frame = apply_clahe(frame)
    img = np.int16(frame)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hsl)

    xval = np.arange(0, 256)
    lut = (255 * np.tanh(vibrance * xval / 255) / np.tanh(1) + 0.5).astype(np.uint8)
    s = cv2.LUT(s, lut)

    h = (h.astype(int) + hue) % 180
    h = h.astype(np.uint8)

    s = cv2.add(s, saturation)
    l = cv2.add(l, lightness)

    # if bilateral_enabled:
        # l = cv2.GaussianBlur(l, (bilateral_sigma_color, bilateral_sigma_space), bilateral_diameter)
        # l = cv2.bilateralFilter(l, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)

    adjusted_hsl = cv2.merge([h, l, s])
    adjusted_frame = cv2.cvtColor(adjusted_hsl, cv2.COLOR_HLS2BGR)

    return adjusted_frame

def preprocLightnessCLAHEImgs(datapath, toSavePath):
    if not os.path.exists(toSavePath):
        os.makedirs(toSavePath)

    # take only images that ends with 3.0.png

    imgs = [img for img in os.listdir(datapath) if img.endswith("3.0.png")]
    for img in tqdm(imgs):
        img_path = os.path.join(datapath, img)
        frame = cv2.imread(img_path)
        adjusted_frame = apply_adjustments(frame)
        adjusted_name = f"{img.split('-')[0]}.png"
        cv2.imwrite(os.path.join(toSavePath, adjusted_name), adjusted_frame)
        # print(f"Saved {img}")


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # datapath = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Test/LLLR"))
    datapath = os.path.normpath("D:\DataSets\!!UPscalinbg\RELLISUR-Dataset\Test\LLLR")
    toSavePath = os.path.normpath(os.path.join(datapath, "../LLLR-CLAHE"))
    preprocLightnessCLAHEImgs(datapath, toSavePath)

