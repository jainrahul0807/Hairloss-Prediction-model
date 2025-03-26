import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import os
from PIL import Image

# Set device to CPU (Change to CUDA if you have a GPU)
device = torch.device("cpu")

# Load trained U-Net model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load("models/best_segmentation_model.pth", map_location=device))
model.eval()
model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Paths
test_image_dir = "dataset/images/test"
output_mask_dir = "dataset/masks/predicted"
os.makedirs(output_mask_dir, exist_ok=True)

# Process test images
test_images = os.listdir(test_image_dir)

for img_name in test_images:
    img_path = os.path.join(test_image_dir, img_name)
    image = Image.open(img_path).convert("RGB")
    image = image.resize((256, 256), Image.BILINEAR)  # Resize for consistency

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict segmentation mask
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy().squeeze()  # Convert to NumPy

    # Convert to binary mask (Threshold = 0.5)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Save predicted mask
    mask_filename = img_name
    mask_path = os.path.join(output_mask_dir, mask_filename)
    cv2.imwrite(mask_path, mask)

    print(f"✅ Saved Predicted Mask: {mask_path}")

print("🎉 Segmentation Evaluation Completed!")
