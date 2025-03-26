import json
import os
import numpy as np
import cv2

# Paths
json_path = "dataset/annotations/instances_val.json"
image_dir = "dataset/images/val/"
mask_dir = "dataset/masks/val/"

# Ensure mask directory exists
os.makedirs(mask_dir, exist_ok=True)

# Load COCO annotations
with open(json_path, "r") as f:
    data = json.load(f)

# Map image IDs to filenames
image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

# Process each annotation
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    segmentation = annotation["segmentation"]
    bbox = annotation.get("bbox", [])

    img_filename = image_id_to_filename.get(image_id)

    if not img_filename:
        print(f"⚠️ No image filename for ID: {image_id}")
        continue

    # Construct correct image path
    img_path = os.path.join(image_dir, img_filename)

    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        continue

    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read image: {img_path}")
        continue

    height, width = img.shape[:2]

    # Create an empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Convert polygon segmentation to OpenCV format
    if segmentation:
        for seg in segmentation:
            poly = np.array(seg, np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [poly], 255)  # White area for segmentation

    # Apply bounding box if available
    if bbox:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)  # Filled rectangle

    # Ensure mask directory structure
    subdir = os.path.dirname(img_filename)
    mask_subdir = os.path.join(mask_dir, subdir)
    os.makedirs(mask_subdir, exist_ok=True)

    # Save mask
    mask_filename = os.path.basename(img_filename)
    mask_path = os.path.join(mask_subdir, mask_filename)

    if not cv2.imwrite(mask_path, mask):
        print(f"❌ Failed to save mask: {mask_path}")

print("✅ Segmentation masks generated successfully!")
