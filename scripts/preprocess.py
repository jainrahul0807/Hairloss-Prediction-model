import json
import pandas as pd

# File paths
train_json_path = "dataset/annotations/instances_train.json"
val_json_path = "dataset/annotations/instances_val.json"
train_csv_path = "dataset/csv/train.csv"
val_csv_path = "dataset/csv/val.csv"

def convert_coco_to_csv(json_file, csv_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    annotations = []
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_file = next(img["file_name"] for img in data["images"] if img["id"] == img_id)
        label = ann["attributes"]["Severity"]  # Extract Norwood Scale class
        annotations.append((img_file, label))

    # Save as CSV
    df = pd.DataFrame(annotations, columns=["image", "label"])
    df.to_csv(csv_file, index=False)
    print(f"✅ Converted {json_file} to {csv_file}")

# Convert files
convert_coco_to_csv(train_json_path, train_csv_path)
convert_coco_to_csv(val_json_path, val_csv_path)
