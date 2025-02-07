import os
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
# from srresnet import _NetG
from custom_srresnet import _NetX2, _NetX2Eff

# Fix memory fragmentation issues
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["OMP_NUM_THREADS"] = "1"  # Prevents CPU threading issues
os.environ["TORCH_HOME"] = r"torch_cache"


current_directory = os.path.dirname(os.path.abspath(__file__))
train_datapathLR = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Train/NLHR/X1")) # Train
train_datapathX2 = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Train/NLHR/X2")) # Train
valid_datapathLR = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Val/NLHR/X1"))
valid_datapathX2 = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Val/NLHR/X2"))
train_datapathLRLL = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Train/LLLR"))
valid_datapathLRLL = os.path.normpath(os.path.join(current_directory, "../RELLISUR-Dataset/Val/LLLR"))


BATCH_SIZE = 6
LEARNING_RATE = 1e-4
EPOCHS = 50
STEP_DECAY = 150  # 200
num_blocks = 1  # 2 default for bottleneck
num_channels = 48  # 32 was default for bottleneck
SAVE_PATH = f"model/SR_EffBottleRes-BS{BATCH_SIZE}-EP{EPOCHS}-B{num_blocks}-Ch{num_channels}.pth"
# model_chpoint_path = "model/srbottle_resnet-BS4-EP40.pth"

class SRDataset(Dataset):
    def __init__(self, lr_folder, hr_folder):
        self.lr_images = sorted(os.listdir(lr_folder))
        self.hr_images = sorted(os.listdir(hr_folder))
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.transform = transforms.Compose([transforms.ToTensor()])
        '''
        self.hr_transform = transforms.Compose([
            transforms.Resize((1250 * upscale_factor, 1250 * upscale_factor), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        '''

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


class SRDatasetLL(Dataset):
    def __init__(self, lr_folder, hr_folder, param=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.param = param  # Parameter value to filter LR images exposure setting
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Group LR images by base name
        self.lr_image_groups = self._group_lr_images()

        # List of HR images
        self.hr_images = sorted(os.listdir(hr_folder))

    def _group_lr_images(self):
        """Group LR images by base name (excluding the '-{param}' part)."""
        lr_images = sorted(os.listdir(self.lr_folder))
        grouped = {}
        for lr_image in lr_images:
            base_name = lr_image.split("-")[0]
            if base_name not in grouped:
                grouped[base_name] = []
            grouped[base_name].append(lr_image)
        return grouped

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_name = self.hr_images[idx]  # HR image name
        base_name = os.path.splitext(hr_name)[0]  # Get base name without extension

        # Find corresponding LR image with specified parameter
        lr_images = self.lr_image_groups.get(base_name, [])
        if self.param is not None:
            lr_image_name = next(
                (img for img in lr_images if f"-{self.param}" in img),
                None,
            )
            if lr_image_name is None:
                raise ValueError(
                    f"No LR image found for HR image '{hr_name}' with param '{self.param}'"
                )
        else:
            # Default to the first LR image if no param is specified
            lr_image_name = lr_images[0]

        lr_path = os.path.join(self.lr_folder, lr_image_name)
        hr_path = os.path.join(self.hr_folder, hr_name)

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        lr_tensor = self.transform(lr_image)
        hr_tensor = self.transform(hr_image)

        return lr_tensor, hr_tensor


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    backend = "gloo" if os.name == "nt" else "nccl"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed every `STEP_DECAY` epochs"""
    lr = LEARNING_RATE * (0.1 ** (epoch // STEP_DECAY))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(rank, world_size):
    """Training using DDP for parallelization"""
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    train_dataset = SRDataset(train_datapathLR, train_datapathX2)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)# , shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler)  # , num_workers=4, pin_memory=True)

    # model = _NetX2(num_blocks, num_channels).to(device)
    model = _NetX2Eff(num_blocks, num_channels).to(device)
    '''checkpoint = torch.load("model/model_srresnet.pth", map_location=device)
    state_dict = checkpoint["model"].state_dict() if "model" in checkpoint else checkpoint
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()}, strict=False)'''

    # checkpoint = torch.load(model_chpoint_path, map_location=device)
    # state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    # filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    # model.load_state_dict(filtered_state_dict, strict=False)
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss(reduction="mean")   # sum # `size_average=False` equivalent
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        start_epoch_time = time.time()
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        model.train()
        total_loss = 0

        scaler = torch.cuda.amp.GradScaler()

        for iteration, batch in enumerate(train_loader, 1):
            lr, hr = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            # print(f"LR Shape: {lr.shape}, HR Shape: {hr.shape}")
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)/ BATCH_SIZE  # Normalize loss per image
            loss.backward()
            optimizer.step()
            '''with torch.cuda.amp.autocast():  # Run forward pass with mixed precision
                output = model(lr)
                loss = criterion(output, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                        torch.cuda.empty_cache()

            '''
            total_loss += loss.item()

            if iteration % 100 == 0:
                print(f"Rank [{rank}] Epoch [{epoch}/{EPOCHS}] Iteration [{iteration}/{len(train_loader)}] - Loss: {loss.item():.5f}")
        avg_train_loss = total_loss / len(train_loader)
        print(f"Rank [{rank}] Epoch [{epoch}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f}")

        dist.barrier()
        # Save Model (Only Rank 0)
        if rank == 0:
            torch.save({"epoch": epoch, "model": model.module.state_dict()}, SAVE_PATH)
            end_epoch_time = time.time()
            print("Epoch training took ", round((end_epoch_time - start_epoch_time)/60, 2), " min")


    cleanup_ddp()


if __name__ == "__main__":
    WORLD_SIZE = 4 # torch.cuda.device_count()
    if WORLD_SIZE < 4:
        raise RuntimeError(f"Need at least 4 GPUs, but found {WORLD_SIZE}")

    mp.set_start_method("spawn", force=True)
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)


