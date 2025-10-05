import os
import re
import csv

# Path to your dataset
dataset_dir = "/Users/siagarg/PycharmProjects/StormHacks OpenCV/ExtDataset"

# Output CSV file
output_csv = "auto_labels.csv"

# Regex to extract label from filename (e.g., "cardboard1.jpg" → "cardboard")
label_pattern = re.compile(r"([a-zA-Z_]+)")

rows = []

for root, dirs, files in os.walk(dataset_dir):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            match = label_pattern.match(filename)
            if match:
                label = match.group(1).lower()
            else:
                label = "unknown"

            image_path = os.path.join(root, filename)
            rows.append((image_path, label))

#Save results
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(rows)

print(f"Auto-labeling complete! Found {len(rows)} images.")
print(f"Labels saved to: {os.path.abspath(output_csv)}")
