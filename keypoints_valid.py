"""
################################################################################
Description:
    This script processes MediaPipe pose keypoint data stored in CSV files and 
    corresponding images. It performs several checks on the extracted keypoints 
    to determine if the pose is valid and suitable for further analysis.

Key Functions:
    - Parses keypoint data from CSV rows.
    - Performs filtering based on:
        - Shoulder width (narrow_torso)
        - Distance from nose to heels (total_fail)
        - Hands being too close to feet (hands_too_close_feet)
    - Computes mean positions of key regions (torso, hands, feet).
    - Applies a simple skin detection filter to ensure keypoints fall on skin pixels.
    - Moves valid annotated images to a separate folder and draws keypoints.
    - Updates the original CSV file with new keypoint coordinates and flags for valid rows.

Output:
    - Annotated valid images saved to `valid_detection` and images with the final 5 points to `valid_points` subfolders.
    - Original CSV is updated with extra columns for valid point coordinates.

Usage:
- Configure `baby_folder` with the root directory of the dataset.
- Optionally set `subfolders` manually for individual debugging.

################################################################################
"""


import math
import csv
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from pathlib import Path



POSE_LANDMARKS = {
    0: 'nose',
    1: 'right_eye_inner',
    2: 'right_eye',
    3: 'right_eye_outer',
    4: 'left_eye_inner',
    5: 'left_eye',
    6: 'left_eye_outer',
    7: 'right_ear',
    8: 'left_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'right_shoulder',
    12: 'left_shoulder',
    13: 'right_elbow',
    14: 'left_elbow',
    15: 'right_wrist',
    16: 'left_wrist',
    17: 'right_pinky',
    18: 'left_pinky',
    19: 'right_index',
    20: 'left_index',
    21: 'right_thumb',
    22: 'left_thumb',
    23: 'right_hip',
    24: 'left_hip',
    25: 'right_knee',
    26: 'left_knee',
    27: 'right_ankle',
    28: 'left_ankle',
    29: 'right_heel',
    30: 'left_heel',
    31: 'right_foot_index',
    32: 'left_foot_index'
}

def compute_distance(p1, p2):
    """
    Compute Euclidean distance between two points. Each point is a tuple (x, y).
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def parse_keypoints(row):
    keypoints = {}
    data = row
    for i in range(33):
        base_idx = 4 + i * 5
        if base_idx + 4 >= len(data):
            continue  # skip if incomplete data
        keypoints[POSE_LANDMARKS[i]] = {
            'x': float(data[base_idx]),
            'y': float(data[base_idx+1]),
            'z': float(data[base_idx+2]),
            'visibility': float(data[base_idx+3]),
            'presence': float(data[base_idx+4])
        }
    return keypoints

def narrow_torso(keypoints):
    left_shoulder_keypoints = keypoints['left_shoulder']
    right_shoulder_keypoints = keypoints['right_shoulder']

    ls_x = left_shoulder_keypoints['x']
    ls_y = left_shoulder_keypoints['y']    

    rs_x = right_shoulder_keypoints['x']
    rs_y = right_shoulder_keypoints['y']

    #calculate distance between left and right shoulder
    shoulder_distance = compute_distance((ls_x, ls_y), (rs_x, rs_y))
    if shoulder_distance < 300:
        return True

    return False

def total_fail(keypoints):
    nose_keypoints = keypoints['nose']
    left_heel_keypoints = keypoints['left_heel']
    right_heel_keypoints = keypoints['right_heel']

    n_x = nose_keypoints['x']
    n_y = nose_keypoints['y']

    lh_x = left_heel_keypoints['x']
    lh_y = left_heel_keypoints['y']

    rh_x = right_heel_keypoints['x']
    rh_y = right_heel_keypoints['y']

    distance_n_lh = compute_distance((n_x, n_y), (lh_x, lh_y))
    distance_n_rh = compute_distance((n_x, n_y), (rh_x, rh_y))
    if distance_n_lh < 1000 or distance_n_rh < 1000:
        return True
    return False

def hands_too_close_feet(keypoints):
    right_hand_keypoint = keypoints['right_wrist']
    left_hand_keypoint = keypoints['left_wrist']
    left_heel_keypoints = keypoints['left_heel']
    right_heel_keypoints = keypoints['right_heel']

    rw_x = right_hand_keypoint['x']
    rw_y = right_hand_keypoint['y']

    lw_x = left_hand_keypoint['x']
    lw_y = left_hand_keypoint['y']

    lh_x = left_heel_keypoints['x']
    lh_y = left_heel_keypoints['y']

    rh_x = right_heel_keypoints['x']
    rh_y = right_heel_keypoints['y']

    distance_lw_lh = compute_distance((lw_x, lw_y), (lh_x, lh_y))
    distance_rw_rh = compute_distance((rw_x, rw_y), (rh_x, rh_y))
    if distance_lw_lh < 50 or distance_rw_rh < 50:
        return True
    return False

def valid(kp): # Check if keypoint is valid based on visibility and presence
    return kp['visibility'] > 0.7 and kp['presence'] > 0.7

def mean_coords(keypoints, names):
    points = [keypoints[name] for name in names if valid(keypoints[name])] # Filter valid keypoints
    if not points:
        return {'x': None, 'y': None}
    x = sum(p['x'] for p in points) / len(points)
    y = sum(p['y'] for p in points) / len(points)
    return {'x': round(x), 'y': round(y)}


def skin_color(keypoints, image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # Convert to YCrCb color space
    h, w, _ = imageYCrCb.shape

    min_YCrCb = np.array([0, 133, 77], np.uint8) # Minimum YCrCb values for skin detection
    max_YCrCb = np.array([235, 173, 127], np.uint8) # Maximum YCrCb values for skin detection
    
    for point in keypoints:
        x = point['x']
        y = point['y']
        if x is not None and y is not None:
            if 0 <= x < w and 0 <= y < h:
                pixel = imageYCrCb[y, x]
                is_skin = np.all(pixel >= min_YCrCb) and np.all(pixel <= max_YCrCb)

                if not is_skin:
                    point['x'] = None
                    point['y'] = None
                    print('not skin pixel')
    return keypoints

# input folder containing the CSV files and images
baby_folder = Path("/home/cvmsct05/temperatures_max_min")

subfolders = [f for f in baby_folder.iterdir() if f.is_dir()] 
for folder in subfolders:
    out = Path(folder)
    print(out)
    csv_files = list(folder.rglob("*.csv"))

    csv_file = csv_files[0]
    output = out / "valid_detection"
    output_valid_points = out / "valid_points"

    rows = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]

    process = headers.index("Process")

    keypoint_headers = ["torso_x", "torso_y", "left_hand_x", "left_hand_y", "right_hand_x", "right_hand_y", "left_foot_x", "left_foot_y", "right_foot_x", "right_foot_y"]
    if "torso_x" not in headers:
        headers.extend(keypoint_headers)

    print('row interation')
    for row in rows:
        if 'True' in row[process]:
            keypoints = parse_keypoints(row)
            if len(keypoints) > 0 :
            
                process_value = row[process]    #process flag

                # Check if the keypoints are valid
                if narrow_torso(keypoints) == False and total_fail(keypoints) == False and hands_too_close_feet(keypoints) == False:
                    torso = mean_coords(keypoints, ['left_shoulder', 'right_shoulder'])
                    left_hand = mean_coords(keypoints, ['left_wrist', 'left_pinky', 'left_index', 'left_thumb'])
                    right_hand = mean_coords(keypoints, ['right_wrist', 'right_pinky', 'right_index', 'right_thumb'])
                    left_foot = mean_coords(keypoints, ['left_ankle', 'left_heel', 'left_foot_index'])
                    right_foot = mean_coords(keypoints, ['right_ankle', 'right_heel', 'right_foot_index'])
                    points = [torso, left_hand, right_hand, left_foot, right_foot]

                    image_path = row[1].strip()
                    image_name = os.path.basename(image_path)
                    image_name, ext = os.path.splitext(image_name)

                    base, ext = os.path.splitext(image_path)
                    vis_path_underscore = f"{base}_VIS{ext}"
                    vis_path_dot = f"{base}.VIS{ext}"

                    # Check which one exists
                    if os.path.exists(vis_path_underscore):
                        result_path = vis_path_underscore
                        image_name = image_name + '_VIS_keypoints.jpg'
                    elif os.path.exists(vis_path_dot):
                        result_path = vis_path_dot
                        image_name = image_name + '.VIS_keypoints.jpg'
                    else:
                        result_path = None

                    #Move the image to an output subfolder
                    if result_path:
                        # Check if the image file exists
                        if os.path.exists(result_path):
                            # Create the output directory if it doesn't exist
                            os.makedirs(output, exist_ok=True)
                            # Move the image to the output directory
                            input_filename = os.path.join(out, image_name)
                            new_image_path = os.path.join(output, image_name)
                            try:
                                os.rename(input_filename, new_image_path)
                                print(f"Moved image to {new_image_path} ")
                            except FileNotFoundError: 
                                print("Not found, skipping")
                        else:
                            print(f"Image file {result_path} does not exist.")

                    #take out '_keypoints' from the result_path
                    if result_path:
                        result_path = result_path.replace('_keypoints', '')
                        result_path = os.path.join(base, result_path)

                    # Output result
                    if result_path:
                        os.makedirs(output_valid_points, exist_ok=True)
                        image = Image.open(result_path)

                        # Draw keypoints on the image
                        draw = ImageDraw.Draw(image)
                        points = skin_color(points, image)

                        color = ['red', 'blue', 'blue', 'green', 'green']
                        for i in range (len(points)):
                            if points[i]['x'] is not None and points[i]['y'] is not None:
                                draw.ellipse((points[i]['x'] - 15, points[i]['y'] - 15, points[i]['x'] + 15, points[i]['y'] + 15), fill=color[i])

                        # Save the image with keypoints drawn
                        output_path = os.path.join(output_valid_points, image_name)
                        image.save(output_path)
                        print(f"Saved image with keypoints to {output_path}")
                    
                    # Update the row with the new keypoint coordinates
                    for p in points:
                        row.append(p['x'] if p['x'] is not None else 'None')
                        row.append(p['y'] if p['y'] is not None else 'None')
                else:
                    row[process] = False

            #If detection is not valid, set process to false
            else:
                row[process] = False

    # print('updating csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

print(f"✅ Keypoints valid done {baby_folder}")
