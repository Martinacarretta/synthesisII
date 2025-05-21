import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image

# Parameters
csv_path = '/home/cvmsct05/temperatures_max_min/29/14.08.24-15.08.24/14.08.24-15.08.24.csv'
frame_duration = 1  # seconds per frame

# Build output paths
csv_parts = csv_path.split('/')
date_folder = csv_parts[-2]
experiment_id = csv_parts[-3]
output_dir = f"/home/cvmsct05/others/gifs/{experiment_id}"
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, f"{date_folder}.mp4")
frames_dir = os.path.join(output_dir, "frames")  # Directory to save individual frames
os.makedirs(frames_dir, exist_ok=True)  # Create the frames directory

# Load CSV
df = pd.read_csv(csv_path)
df = df[df['Process'] == True]

# Keypoints
points = {
    'torso': ('torso_x', 'torso_y', 'torso_temp'),
    'left_hand': ('left_hand_x', 'left_hand_y', 'left_hand_temp'),
    'right_hand': ('right_hand_x', 'right_hand_y', 'right_hand_temp'),
    'left_foot': ('left_foot_x', 'left_foot_y', 'left_foot_temp'),
    'right_foot': ('right_foot_x', 'right_foot_y', 'right_foot_temp'),
}

# Helper: Load VIS image with PIL
def load_vis_image(base_path):
    root, ext = os.path.splitext(base_path)
    candidates = [
        f"{root}_VIS{ext}",
        f"{root}.VIS{ext}",
        f"{root}_VIS{ext.lower()}",
        f"{root}.VIS{ext.lower()}"
    ]
    for vis_path in candidates:
        if os.path.exists(vis_path):
            try:
                img = Image.open(vis_path).convert('RGB')  # open and ensure RGB
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # convert to BGR
                return img_bgr
            except Exception as e:
                print(f"Failed loading {vis_path}: {e}")
    return None

# Collect frames
frames = []
for idx, row in df.iterrows():
    temps = [row.get(f"{p}_temp", np.nan) for p in points.keys()]
    if all(pd.isna(t) for t in temps):
        print("image skipped, no temps")
        continue  # Skip if all temperatures are NaN

    img_path = row['Image Path'].strip()
    vis_img = load_vis_image(img_path)
    if vis_img is None:
        continue  # Skip if no VIS image found

    # Draw keypoints
    for label, (x_col, y_col, temp_col) in points.items():
        x = row.get(x_col, np.nan)
        y = row.get(y_col, np.nan)
        temp = row.get(temp_col, np.nan)

        if not np.isnan(x) and not np.isnan(y):
            center = (int(x), int(y))
            cv2.circle(vis_img, center, radius=15, color=(0, 255, 0), thickness=-1)  # bigger green dot
            temp_text = f"{label}: {temp:.1f}" if not pd.isna(temp) else f"{label}: nan"
            
            # Add a translucent background rectangle for better text visibility
            (text_width, text_height), _ = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(vis_img, 
                            (center[0]+10, center[1]),
                            (center[0]+10 + text_width+20, center[1]-30 - text_height-10), (0, 0, 255), -1)  # semi-transparent red rectangle
            
            # Add text (with thicker font for bold effect)
            cv2.putText(vis_img, temp_text, (center[0]+10, center[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Extract image path containing 'HM...' part
    image_path_parts = img_path.split('/')
    for part in image_path_parts:
        if 'HM' in part:
            image_name = part
            break
    else:
        image_name = "Unknown"  # In case no HM part is found
    
    # Display image path (just HM part)
    height, width, _ = vis_img.shape
    text_size = cv2.getTextSize(image_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = int((width - text_size[0]) / 2)  # Calculate x position for center alignment
    text_y = 40  # Fixed y position at the top

    # Add a translucent background rectangle for the image name
    cv2.rectangle(vis_img, (text_x - 10, text_y - 25),
                    (text_x + text_size[0] + 10, text_y + text_size[1] + 5), (0, 0, 255), -1)  # background

    # Display image name at the top center
    cv2.putText(vis_img, image_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    frames.append(vis_img)

    # Save each frame as a PNG image
    frame_filename = f"frame_{idx:04d}.png"  # Use index for filename
    cv2.imwrite(os.path.join(frames_dir, frame_filename), vis_img)
    print(f"Saved frame: {frame_filename}")

# Save video
if frames:
    height, width, _ = frames[0].shape
    fps = 1 / frame_duration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_video_path}")
else:
    print("No valid frames found!")

print(f"Total frames saved: {len(frames)}")
