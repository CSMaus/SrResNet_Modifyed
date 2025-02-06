import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from srresnet import _NetG

# Dataset paths
current_directory = os.path.dirname(os.path.abspath(__file__))
train_datapathLR = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Train/NLHR/X1"))
train_datapathX2 = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Train/NLHR/X2"))
valid_datapathLR = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Val/NLHR/X1"))
valid_datapathX2 = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Val/NLHR/X2"))

# Training Parameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
SAVE_PATH = "model/srresnet_finetuned.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom Dataset
class SRDataset(Dataset):
    def __init__(self, lr_folder, hr_folder):
        self.lr_images = sorted(os.listdir(lr_folder))
        self.hr_images = sorted(os.listdir(hr_folder))
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_folder, self.lr_images[idx])
        hr_path = os.path.join(self.hr_folder, self.hr_images[idx])

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        lr_tensor = self.transform(lr_image)
        hr_tensor = self.transform(hr_image)

        return lr_tensor, hr_tensor


if __name__ == "__main__":
    # Load Datasets
    train_dataset = SRDataset(train_datapathLR, train_datapathX2)
    valid_dataset = SRDataset(valid_datapathLR, valid_datapathX2)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)  # FIXED: num_workers=0 for Windows
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)  # FIXED: num_workers=0

    # Load Model
    model = _NetG()
    checkpoint = torch.load("model/model_srresnet.pth", map_location=device)
    state_dict = checkpoint["model"].state_dict() if "model" in checkpoint else checkpoint
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for lr, hr in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for lr, hr in valid_loader:
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                loss = criterion(sr, hr)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save model checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict()}, SAVE_PATH)

    print("Fine-tuning complete. Model saved.")
