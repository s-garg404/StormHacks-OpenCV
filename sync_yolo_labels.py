import pandas as pd
import os

# Paths
dataset_dir = "/Users/siagarg/PycharmProjects/StormHacks OpenCV/ExtDataset"
csv_file = os.path.join(dataset_dir, "auto_labels.csv")
labels_dir = os.path.join(dataset_dir, "labels")

# Make labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Define class names in the order you want YOLO to see them
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# Load CSV
df = pd.read_csv(csv_file)

updated_files = 0
for _, row in df.iterrows():
    # Resolve image path relative to dataset_dir
    image_path = os.path.join(dataset_dir, row['image_path'])
    if not os.path.isfile(image_path):
        print(f"Warning: image not found: {image_path}")
        continue

    label_name = row['label']
    if label_name not in class_to_idx:
        print(f"Warning: unknown label '{label_name}' for image {image_path}")
        continue

    # Prepare YOLO label file path
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_file = os.path.join(labels_dir, f"{image_name}.txt")

    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
    # Here we just use a dummy full-image box: 0.5 0.5 1.0 1.0
    class_id = class_to_idx[label_name]
    with open(txt_file, "w") as f:
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    updated_files += 1

print(f"Done syncing labels. Created/updated {updated_files} files.")
