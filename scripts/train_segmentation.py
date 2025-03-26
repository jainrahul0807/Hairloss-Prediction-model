import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from PIL import Image

# Set device to CPU (Change to CUDA if you have a GPU)
device = torch.device("cpu")

# Custom Dataset for Scalp Segmentation
class ScalpSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale mask

        # Resize images & masks
        image = image.resize((256, 256), Image.BILINEAR)  # Resize images
        mask = mask.resize((256, 256), Image.NEAREST)  # Resize masks

        # Convert mask to binary (0 for hair, 1 for bald)
        # mask = np.array(mask) / 255.0
        # mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize mask

        return image, mask

# Define Image Transformations
transform = transforms.Compose([
    # transforms.Resize((256,256)),  # Resize for CPU efficiency
    transforms.ToTensor(),
])

# Load Datasets
train_dataset = ScalpSegmentationDataset("dataset/images/train/bald", "dataset/masks/train/bald", transform)
val_dataset = ScalpSegmentationDataset("dataset/images/val/bald", "dataset/masks/val/bald", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# Define U-Net Model
model = smp.Unet(
    encoder_name="resnet18",  # Backbone (efficient for CPU)
    encoder_weights="imagenet",  # Pre-trained on ImageNet
    in_channels=3,
    classes=1,  # Binary classification (bald vs. hair)
    activation=None
)

model.to(device)

# Define Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary segmentation loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "models/best_segmentation_model.pth")
print("✅ U-Net Segmentation Model Trained & Saved!")
