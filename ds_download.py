import os
import requests
from tqdm import tqdm

train_dir = "D:\ML_DL_AI_stuff\!!DoosanWelding\Data\HQ-50K\HQ-50K//train"
test_dir = "D:\ML_DL_AI_stuff\!!DoosanWelding\Data\HQ-50K\HQ-50K//test"
# os.makedirs(output_dir, exist_ok=True)

def download_images(folder, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    txt_file = os.path.join(folder, "all.txt")
    with open(txt_file, "r") as f:
        urls = f.readlines()
    for url in tqdm(urls, desc=f"Downloading {folder}"):
        url = url.strip()
        try:
            response = requests.get(url, stream=True, timeout=10)
            if response.status_code == 200:
                filename = os.path.join(output_subdir, os.path.basename(url))
                with open(filename, "wb") as img_file:
                    for chunk in response.iter_content(1024):
                        img_file.write(chunk)
        except Exception:
            continue

download_images(train_dir, os.path.join(train_dir))
download_images(test_dir, os.path.join(test_dir))