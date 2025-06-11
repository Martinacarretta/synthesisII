"""
================================================================================
Pose keypoint extractor for newborn images using MediaPipe

Description:
    This script processes folders of newborn images to extract pose keypoints
    using MediaPipe's Pose Landmarker. It reads corresponding CSV files to
    check which images to process, appends keypoint data (X, Y, Z, Visibility,
    Presence) to the CSV, and saves annotated images with detected landmarks.

Usage:
    - Set the `input_folder` to the directory containing image subfolders.
    - Set the `output_root` to the corresponding result directory containing CSVs.
    - Ensure the `pose_landmarker.task` model file is available.

Output:
    - Updated CSV files with 165 pose-related columns (5 values x 33 landmarks).
    - Annotated images showing pose detection results or "no detection" images.

================================================================================
"""

print('importing libraries')
import csv
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print('libraries imported successfully')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_tracking_confidence=0.8)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    mp_drawing.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),  # thickness & radius
      mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
      )
  return annotated_image

# Read the file contents
with open(r"/home/cvmsct05/pose_landmarker.task", "rb") as f:  
    model_buffer = f.read()

base_options = python.BaseOptions(model_asset_buffer=model_buffer)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)

detector = vision.PoseLandmarker.create_from_options(options)


def process_folder(folder_path, output_subfolder):
    #find the csv file in the folder
    csv_files = [f for f in os.listdir(output_subfolder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV file found in {output_subfolder}. Exiting.")
        return
    csv_file = csv_files[0]
    csv_file = os.path.join(output_subfolder, csv_file)

    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        if "X_0" in headers:
            print(f"Keypoint data already present in {csv_file}. Skipping folder {folder_path}.")
            return
    
    # create header for the keypoints
    max_landmarks = 33  
    keypoint_headers = [f"{coord}_{i}" for i in range(max_landmarks) for coord in ["X", "Y", "Z", "Vis", "Pres"]]
    
    #use 'normal' images only
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                   if "VIS" in file and file.lower().endswith((".jpg", ".jpeg", ".png")) and 'keypoints' not in file]

    # Read existing CSV into a list of rows
    rows = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]

    # Map image paths for quick lookup
    path_index = headers.index("Image Path")
    process = headers.index("Process")

    # Extend headers if keypoint columns are not yet added
    if "X_0" not in headers:
        headers.extend(keypoint_headers)

    # Process each image
    for image_path in image_paths:
        # print(f"Processing image: {image_path}")
        process_value = None
        matching_row = None
        #extract relative path for comparison
        rel_image_path = os.path.relpath(image_path, folder_path)
        #consider _VIS or .VIS format
        rel_image_path = re.split(r'_VIS|\.VIS', rel_image_path)[0]

        for row in rows:
            if rel_image_path in row[path_index]: # check if the image path matches
                matching_row = row
                process_value = row[process]
                break

        if process_value == 'True':
            if matching_row != None:
                image = mp.Image.create_from_file(image_path)
                width = image.width
                height = image.height
                image_np = image.numpy_view()
                detection_result = detector.detect(image)
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    keypoints = detection_result.pose_landmarks[0]

                    keypoint_values = []
                    for j in range(len(keypoints)):
                        x = keypoints[j].x
                        y = keypoints[j].y
                        x = int(x * width)   ######### normal coordinates
                        y = int(y * height)  #########
                        z = keypoints[j].z
                        visibility = keypoints[j].visibility
                        presence = keypoints[j].presence
                        keypoint_values.extend([x, y, z, visibility, presence])

                    if matching_row:
                        matching_row.extend([str(v) for v in keypoint_values]) # extend the row with keypoint values

                    # Save annotated image
                    annotated_image = draw_landmarks_on_image(image_np, detection_result)
                    annotated_out = os.path.join(output_subfolder, f"{os.path.splitext(os.path.basename(image_path))[0]}_keypoints.jpg")
                    cv2.imwrite(annotated_out, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) #save annotated image
                else:       #If no landmarks
                    print(f"No landmarks detected for {image_path}")
                    #modify the process value to False in the csv
                    if matching_row:
                        matching_row[process] = 'False'

                    original_out = os.path.join(output_subfolder, f"no_detection_{os.path.basename(image_path)}")
                    cv2.imwrite(original_out, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)) #save original image with no detection
            else:                
                continue  # Skip if no matching row found

        else:
            print(f"Skipping {image_path} as process is set to False")
    
    # Write updated rows back to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

# input and output directories
input_folder = r"/data/uabcvmsc/shared/newborn"
output_root = r"/home/cvmsct05/temperatures_max_min"

os.makedirs(output_root, exist_ok=True)

print('Image processing started')
for root, dirs, files in os.walk(input_folder):
    if not dirs:  # Final subfolder (no more subdirectories)
        rel_path = os.path.relpath(root, input_folder)
        output_folder = os.path.join(output_root, rel_path)
        # Check if corresponding folder exists in temperatures_max_min
        if not os.path.exists(output_folder):
            print(f"Skipping {root} because corresponding folder does not exist in temperatures_max_min")
            continue

        process_folder(root, output_folder)

print(f"✅ Processing keypoints complete. Results saved in {output_root}")
