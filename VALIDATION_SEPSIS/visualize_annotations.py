import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Paths to your CSVs
csv_paths = [
    "/data/uabcvmsc/cvmsct05/ValidationManualCsvs/2955_keypoints_with_temps.csv",
    "/data/uabcvmsc/cvmsct05/ValidationManualCsvs/output_keypoints_with_temps.csv",
    "/data/uabcvmsc/cvmsct05/ValidationManualCsvs/outputkeypointsares_correct_paths_with_temps.csv"
]

# Load CSVs
a1 = pd.read_csv(csv_paths[0])
a2 = pd.read_csv(csv_paths[1])
a3 = pd.read_csv(csv_paths[2])

# Extract common image names
common_images = set(a1['Image Name']) & set(a2['Image Name']) & set(a3['Image Name'])
common_images = list(common_images)
np.random.shuffle(common_images)
selected_images = common_images[:50]

# Output directory for visualizations
output_dir = "/data/uabcvmsc/cvmsct05/ValidationManualCsvs/annotated_visuals"
os.makedirs(output_dir, exist_ok=True)

# Define parts and color/marker per annotator
parts = ['torso', 'left_hand', 'right_hand']
colors = ['r', 'g', 'b']
markers = ['x', 'o', 's']
annotators = [a1, a2, a3]

# Loop through selected images
for idx, img_path in enumerate(selected_images):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load {img_path}")
        continue

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='hot')
    plt.title(f"Image {idx+1}")

    for i, df in enumerate(annotators):
        row = df[df['Image Name'] == img_path]
        if row.empty:
            continue
        for part in parts:
            x = row[f'{part}_x'].values[0]
            y = row[f'{part}_y'].values[0]
            if not pd.isna(x) and not pd.isna(y):
                plt.scatter(x, y, c=colors[i], marker=markers[i], label=f'{part} - a{i+1}')
    
    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc='upper right', fontsize=8)
    plt.axis('off')
    plt.tight_layout()

    filename = os.path.basename(img_path).replace(".jpeg", f"_vis_{idx+1}.png")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"\n✅ Saved {len(selected_images)} visual comparisons to: {output_dir}")
