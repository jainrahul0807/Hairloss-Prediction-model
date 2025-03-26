import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
import torchvision.models as models
# Set device to CPU
device = torch.device("cpu")


# Dataset Class
class NorwoodDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {"norwood_1": 0, "norwood_2": 1, "norwood_3": 2, "norwood_4": 3,
                          "norwood_5": 4, "norwood_6": 5, "norwood_7": 6}

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     img_path = os.path.normpath(os.path.join(self.img_dir, self.data.iloc[idx, 0]))
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     label = self.label_map[self.data.iloc[idx, 1]]
    #     return image, label
    def __getitem__(self, idx):
        img_path = os.path.normpath(os.path.join(self.img_dir, self.data.iloc[idx, 0]))

        # Debug: Print the image path
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")

        # image = cv2.imread(img_path)
        # if image is None:
        #     print(f"❌ OpenCV could not read the image: {img_path}")
        #
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Error happens here if image is None
        image=Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.label_map[self.data.iloc[idx, 1]]
        return image, label


# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Reduce image size for CPU efficiency
    transforms.RandomHorizontalFlip(p=0.5),  # Flip image horizontally
    transforms.RandomRotation(10),  # Rotate by ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Ad
    transforms.ToTensor(),
])

# Load Data
train_dataset = NorwoodDataset("dataset/csv/train.csv", "dataset/images/train/", transform)
val_dataset = NorwoodDataset("dataset/csv/val.csv", "dataset/images/val/", transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load MobileNetV3 (Lightweight)
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
# model =torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)  # 7 Norwood Scale classes

# Training Parameters
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

if __name__=="__main__":
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=0)
# Training Loop
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "models/best_norwood_model_NET_BO.pth")
print("✅ Norwood Classification Model Trained & Saved!")
