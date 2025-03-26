import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import pandas as pd
from PIL import Image
import os
import torchvision
from torchvision.models import MobileNet_V3_Small_Weights
import torchvision.models as models

# Load Model
device = torch.device("cpu")
# model =torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,7)
model.load_state_dict(torch.load("models/best_norwood_model_NET_BO.pth", map_location=device))
model.eval()

# Load Validation Data
val_csv = "dataset/csv/val.csv"
val_images_path = "dataset/images/val/"
df = pd.read_csv(val_csv)

# Define Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Initialize Lists
y_true = []
y_pred = []
label_map = {"norwood_1": 0, "norwood_2": 1, "norwood_3": 2, "norwood_4": 3,
             "norwood_5": 4, "norwood_6": 5, "norwood_7": 6}
reverse_label_map = {v: k for k, v in label_map.items()}

# Evaluate Model
for _, row in df.iterrows():
    img_path = os.path.join(val_images_path, row["image"])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, 1).item()

    y_true.append(label_map[row["label"]])
    y_pred.append(prediction)

# Generate Classification Report
print(classification_report(y_true, y_pred, target_names=reverse_label_map.values()))
