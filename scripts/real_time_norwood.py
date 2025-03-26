import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from PIL import Image
import torchvision.models as models
import timm  # EfficientNet
import tkinter as tk
from tkinter import filedialog

# Load models
device = torch.device("cpu")

# Load DeepLabV3+ Segmentation Model
unet_model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
unet_model.load_state_dict(torch.load("models/best_segmentation_model.pth", map_location=device))
unet_model.eval().to(device)

# Load EfficientNet for Norwood Classification
# norwood_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=7)
# norwood_model.load_state_dict(torch.load("models/best_norwood_model.pth", map_location=device))
# norwood_model.eval().to(device)

norwood_model=models.mobilenet_v3_small(weights=None)
norwood_model.classifier[3]=torch.nn.Linear(norwood_model.classifier[3].in_features,7)
norwood_model.load_state_dict(torch.load("models/best_norwood_model.pth", map_location=device))
norwood_model.eval().to(device)


# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Norwood Scale Mapping
label_map = {0: "Norwood 1", 1: "Norwood 2", 2: "Norwood 3", 3: "Norwood 4",
             4: "Norwood 5", 5: "Norwood 6", 6: "Norwood 7"}

# Open File Dialog for Image Selection
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
if not image_path:
    print("❌ No image selected. Exiting.")
    exit()

# Load Image
image = Image.open(image_path).convert("RGB")
image_resized = image.resize((256, 256), Image.BILINEAR)

# Apply transformations
input_tensor = transform(image_resized).unsqueeze(0).to(device)

# Predict segmentation mask (bald area)
with torch.no_grad():
    mask_output = unet_model(input_tensor)
    mask = torch.sigmoid(mask_output).cpu().numpy().squeeze()

# Convert mask to binary (threshold = 0.5)
# mask = (mask > 0.5).astype(np.uint8) * 255
# mask_resized = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
# mask_resized = np.expand_dims(mask_resized, axis=-1)
# mask_resized = np.repeat(mask_resized, 3, axis=-1)

mask = (mask > 0.5).astype(np.uint8) * 255
mask_resized=cv2.resize(mask,(image.width,image.height),interpolation=cv2.INTER_NEAREST)
mask_resized=np.expand_dims(mask_resized, axis=-1)
mask_resized=np.repeat(mask_resized,3,axis=-1)

# Extract only bald area from original image
segmented_image = np.array(image) * mask_resized

# Convert segmented image to tensor
segmented_pil = Image.fromarray(segmented_image.astype(np.uint8))
segmented_tensor = transform(segmented_pil).unsqueeze(0).to(device)

# Predict Norwood Scale
with torch.no_grad():
    output = norwood_model(segmented_tensor)
    probabilities = torch.nn.functional.softmax(output / 2.0, dim=1)  # Temperature Scaling
    norwood_class = torch.argmax(probabilities, 1).item()

# Get Norwood Scale Name
norwood_label = label_map[norwood_class]

# Save & Show Results
output_path = "output_segmented.png"
segmented_pil.save(output_path)
print(f"✅ Prediction: {norwood_label}")
print(f"Segmented image saved as {output_path}")


# import torch
# import torchvision.transforms as transforms
# import segmentation_models_pytorch as smp
# import cv2
# import numpy as np
# from PIL import Image
# import torchvision.models as models
# import matplotlib.pyplot as plt
# from sympy.stats.rv import probability
#
# # Load models
# device = torch.device("cpu")
#
# # Load U-Net Segmentation Model
# unet_model = smp.Unet(
#     encoder_name="resnet18",
#     encoder_weights=None,
#     in_channels=3,
#     classes=1,
#     activation=None
# )
# unet_model.load_state_dict(torch.load("models/best_segmentation_model.pth", map_location=device))
# unet_model.eval().to(device)
#
# # Load Norwood Classification Model
# norwood_model=models.mobilenet_v3_small(weights=None)
# norwood_model.classifier[3]=torch.nn.Linear(norwood_model.classifier[3].in_features,7)
# norwood_model.load_state_dict(torch.load("models/best_norwood_model.pth", map_location=device))
# norwood_model.eval().to(device)
#
# # Transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])
#
# # Norwood Scale Mapping
# label_map = {0: "Norwood 1", 1: "Norwood 2", 2: "Norwood 3", 3: "Norwood 4",
#              4: "Norwood 5", 5: "Norwood 6", 6: "Norwood 7"}
#
# # Start Webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert frame to PIL image
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     image_resized = image.resize((256, 256), Image.BILINEAR)
#
#     # Apply transformations
#
#     input_tensor = transform(image_resized).unsqueeze(0).to(device)
#
#     # Predict segmentation mask (bald area)
#     with torch.no_grad():
#         mask_output = unet_model(input_tensor)
#         mask = torch.sigmoid(mask_output).cpu().numpy().squeeze()
#
#     # Convert mask to binary (threshold = 0.5)
#     mask = (mask > 0.5).astype(np.uint8) * 255
#     mask_resized=cv2.resize(mask,(image.width,image.height),interpolation=cv2.INTER_NEAREST)
#     mask_resized=np.expand_dims(mask_resized, axis=-1)
#     mask_resized=np.repeat(mask_resized,3,axis=-1)
#     # Extract only bald area from original image
#     segmented_image = np.array(image) * mask_resized
#
#     # Convert segmented image to tensor
#     segmented_pil = Image.fromarray(segmented_image)
#     segmented_tensor = transform(segmented_pil).unsqueeze(0).to(device)
#
#     # Predict Norwood Scale
#     with torch.no_grad():
#         output = norwood_model(segmented_tensor)
#         probabilities=torch.nn.functional.softmax(output,dim=1)
#         norwood_class = torch.argmax(probabilities, 1).item()
#
#     # Get Norwood Scale Name
#     norwood_label = label_map[norwood_class]
#
#     # Display result on webcam
#     cv2.putText(frame, f"Norwood Scale: {norwood_label}", (30, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     cv2.imshow("Norwood Scale Detection", frame)
#     # frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     # plt.imshow(frame_rgb)
#     # plt.axis("off")
#     # plt.title(f"Norwood Scale: {norwood_label}")
#     # plt.show()
#
#     key=cv2.waitKey(1) & 0xFF
#     if key == ord('q') or cv2.getWindowProperty("Norwood Scale Detection", cv2.WND_PROP_VISIBLE) < 1:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
