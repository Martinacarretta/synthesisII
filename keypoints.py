print('importing libraries')
import os
import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import csv

import re

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
    #solutions.drawing_utils.draw_landmarks(
    mp_drawing.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      #solutions.drawing_styles.get_default_pose_landmarks_style()
      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),  # Increased thickness & radius
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

    #create headers for the keypoints
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

    image_paths = image_paths[:4]  ##################################

    # Process each image
    for image_path in image_paths:
        # print(f"Processing image: {image_path}")
        process_value = None
        matching_row = None
        #extract relative path for comparison
        rel_image_path = os.path.relpath(image_path, folder_path)
        #consider _VIS or .VIS
        #rel_image_path = rel_image_path.split('_VIS')[0] 
        rel_image_path = re.split(r'_VIS|\.VIS', rel_image_path)[0]

        for row in rows:
            if rel_image_path in row[path_index]:
                matching_row = row
                process_value = row[process]
                break

        # print(process_value)
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
                        x = int(x * width)   #########
                        y = int(y * height)  #########
                        z = keypoints[j].z
                        visibility = keypoints[j].visibility
                        presence = keypoints[j].presence
                        keypoint_values.extend([x, y, z, visibility, presence])

                    if matching_row:
                        matching_row.extend([str(v) for v in keypoint_values])
                        print(matching_row)

                    # Save annotated image
                    annotated_image = draw_landmarks_on_image(image_np, detection_result)
                    annotated_out = os.path.join(output_subfolder, f"{os.path.splitext(os.path.basename(image_path))[0]}_keypoints.jpg")
                    cv2.imwrite(annotated_out, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                else:
                    #If no landmarks, optionally note it or skip
                    print(f"No landmarks detected for {image_path}")

                    #modify the process value to False in the csv
                    if matching_row:
                        matching_row[process] = 'False'

                    original_out = os.path.join(output_subfolder, f"no_detection_{os.path.basename(image_path)}")
                    cv2.imwrite(original_out, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            else:                
                continue  # Skip if no matching row found

        else:
            print(f"Skipping {image_path} as process is set to False")
    
    # Write updated rows back to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# Define input and output directories
input_folder = r"/data/uabcvmsc/shared/newborn/40"
output_root = r"/home/cvmsct05/temperatures_max_min/40"


os.makedirs(output_root, exist_ok=True)

print('Image processing started')
for root, dirs, files in os.walk(input_folder):
    if not dirs:  # Final subfolder (no more subdirectories)
        rel_path = os.path.relpath(root, input_folder)
        output_folder = os.path.join(output_root, rel_path)
        # os.makedirs(output_folder, exist_ok=True)
        process_folder(root, output_folder)

print(f"Processing complete. Results saved in {output_root}")
