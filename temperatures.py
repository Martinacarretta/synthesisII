import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
import random
import os
import glob
from PIL import Image, ImageDraw
import pytesseract
import re
import math


####################################### STEP 1: return path of image, min temp, max temp #######################################

def read_file_line (line):
    # Split the line by commas and strip whitespace
    parts = line.strip().split(',')
    
    # Extract the path and temperature values
    process = parts[0].strip()
    path = parts[1].strip()
    maxt = float(parts[2].strip())
    mint = float(parts[3].strip())
    
    #extract keypoints we want the temp of
    keypoints = [(parts[59].strip(), parts[60].strip()),   # Point 11 (X_11, Y_11)
                           (parts[64].strip(), parts[65].strip()),   # Point 12 (X_12, Y_12)
                           (parts[74].strip(), parts[75].strip()),   # Point 15 (X_15, Y_15)
                           (parts[79].strip(), parts[80].strip()),   # Point 16 (X_16, Y_16)
                           (parts[114].strip(), parts[115].strip()), # Point 23 (X_23, Y_23)
                           (parts[119].strip(), parts[120].strip()), # Point 24 (X_24, Y_24)
                           (parts[134].strip(), parts[135].strip()), # Point 27 (X_27, Y_27)
                           (parts[139].strip(), parts[140].strip())  # Point 28 (X_28, Y_28)
                          ]
    
    return process, path, mint, maxt, keypoints
    

####################################### STEP 2: convert to grayscale ##############################################################################

def normalize(value, min_val, max_val, new_min, new_max):
    return new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val)


def color_to_temp_to_grayscale(image, newborn_id, max_temp, min_temp):
    # Define a fine-grained normalized temperature-to-color mapping TO PRESERVE THE ORDER OF THE TEMPERATURE SCALE. 
    # DICTIONARY GOES FROM 0 TO 1 SO THAT WE CAN USE THE MAX AND MIN TEMPERATURE VALUES TO NORMALIZE THE TEMPERATURES FOR EACH PICTURE ACCORDINGLY
    if newborn_id == "29" or newborn_id == "30" or newborn_id == "31" : #landscape
        #print("baby is 29, 30 or 31, using first scale")
        temperature_colors = { 
            # yellow is after white (should be the hottest, so 1.00) 
            # and we go from white, yellow, red, green, blue, purple (coldest) (0.00 in grayscale normalized)
            1.00: (255, 255, 255), # White (hottest)
            0.95: (255, 255, 0),   # Yellow
            0.90: (255, 100, 0),   # Deep Orange
            0.80: (255, 0, 0),     # Red
            0.70: (255, 100, 100), # Light Red
            0.60: (255, 200, 0),   # Dark Yellow/Orange
            0.50: (0, 255, 0),     # Green
            0.40: (0, 128, 255),   # Light Blue
            0.30: (0, 0, 255),     # Blue
            0.20: (0, 0, 200),     # Deep Blue
            0.10: (75, 0, 130),    # Purple/Black (coldest)
            0.00: (0, 0, 0)        # Black (coldest)
        }
        
    else:
        temperature_colors = {
            0.00: (75, 0, 130),    # Purple/Black (coldest)
            0.10: (0, 0, 200),     # Deep Blue
            0.20: (0, 0, 255),     # Blue
            0.30: (0, 128, 255),   # Light Blue
            0.40: (0, 255, 0),     # Green
            0.50: (173, 255, 47),  # Yellow-Green
            0.60: (255, 255, 0),   # Yellow
            0.70: (255, 200, 0),   # Dark Yellow/Orange
            0.75: (255, 165, 0),   # Orange
            0.80: (255, 100, 0),   # Deep Orange
            0.90: (255, 0, 0),     # Red
            0.95: (255, 100, 100), # Light Red
            1.00: (255, 255, 255)  # White (hottest)
        }

    # Convert dictionary to numpy arrays for processing
    normalized_temps = np.array(list(temperature_colors.keys())) # keys
    colors = np.array(list(temperature_colors.values()), dtype=np.uint8) # values
    
    # RGB to LAB because CIEDE2000 is more accurate in the LAB color space
    # LAB: difference between two colors in Lab space corresponds to how humans perceive the difference.
    # deltaE_ciede2000 computes the perceptual diff between two colors in LAB. this is so that we can find the closest color in the dictionary to each pixel
    colors_lab = rgb2lab(colors.reshape(1, -1, 3)).reshape(-1, 3)
    image_reshaped = image.reshape(-1, 3).astype(np.uint8)
    image_lab = rgb2lab(image_reshaped.reshape(1, -1, 3)).reshape(-1, 3)
    
    # split the image into chunks for parallel processing (function below)
    num_chunks = 8
    chunks = np.array_split(image_lab, num_chunks)
    
    def process_chunk(chunk):
        # Reshape chunk and colors_lab for broadcasting
        chunk_expanded = chunk[:, np.newaxis, :]  # Shape: (N_pixels_in_chunk, 1, 3)
        colors_lab_expanded = colors_lab[np.newaxis, :, :]  # Shape: (1, N_colors, 3)
        
        # compute difference for all pixels in the chunk and all colors in the dictionary
        distances = deltaE_ciede2000(chunk_expanded, colors_lab_expanded)  # Shape: (N_pixels_in_chunk, N_colors)
        
        closest_indices = np.argmin(distances, axis=1) # get closest color with the smallest deltaE (difference)
        return normalized_temps[closest_indices]
    
    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk) for chunk in chunks)
    temperature_map = np.concatenate(results) # Concatenate the results
    
    # Reshape back to image dimensions
    temperature_map = temperature_map.reshape(image.shape[:2])
    
    # normalized temperature --> actual temperature range
    actual_temp_values = temperature_map * (max_temp - min_temp) + min_temp
    
    # tewmperature range --> grayscale (0-255) ensuring red (high temp) is white and blue (low temp) is black
    grayscale_image = np.clip((actual_temp_values - min_temp) / (max_temp - min_temp) * 255, 0, 255).astype(np.uint8)
    
    return grayscale_image


####################################### STEP 3: check the temperatures of the keypoints #########################################################

def map_normal_to_cropped(baby_ID, x, y, normal_width, normal_height, normalized=True):
    if normalized:
        x = int(round(x * normal_width))
        y = int(round(y * normal_height))
        
    if baby_ID == '29' or baby_ID =='30' or baby_ID =='31':
        #print('using rotated crop of 29/30/31')
        # Step 1: Rotate 90° counterclockwise
        x1 = y
        y1 = normal_width - 1 - x

        # Crop bounds (from rotated image)
        x_start, y_start = 430, 450
        x_end, y_end = 2140, 2700

        # Step 2: Check if point is in crop region
        if not (x_start <= x1 < x_end and y_start <= y1 < y_end):
            return None, None  # Point not in cropped region

        # Step 3: Adjust to cropped coordinates
        x2 = x1 - x_start
        y2 = y1 - y_start

        # Step 4: Resize to (480, 640)
        scale_x = 480 / (x_end - x_start)  # 480 / 1710
        scale_y = 640 / (y_end - y_start)  # 640 / 2250

        x3 = x2 * scale_x
        y3 = y2 * scale_y

        # Step 5: Rotate 90° clockwise
        x4 = int(round(639 - y3))
        y4 = int(round(x3))
        
    else:
        #print("Using others")
        # Standard crop (no rotation), cropped to (640, 480)
        x_start, y_start = 485, 360
        x_end, y_end = 2785, 2010

        if not (x_start <= x < x_end and y_start <= y < y_end):
            return None, None  # Point not in cropped region

        # Translate to cropped image coordinates
        x_cropped = x - x_start
        y_cropped = y - y_start

        # Resize to (640, 480)
        scale_x = 640 / (x_end - x_start)  # 640 / 2300
        scale_y = 480 / (y_end - y_start)  # 480 / 1650

        x_resized = int(round(x_cropped * scale_x))
        y_resized = int(round(y_cropped * scale_y))

        x4 = x_resized
        y4 = y_resized

    return x4, y4

def check_temperatures(path, newborn_id, max_temp, min_temp, keypoints):
        try:
            pil_image = Image.open(path) #thermal
            image = pil_image.convert("RGB")

            # Convert PIL Image to numpy array
            image = np.array(image)

        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            return [float('nan')] * len(keypoints)
        
        
        grayscale_image = color_to_temp_to_grayscale(image, newborn_id, max_temp, min_temp) # RGB --> grayscale (with temperature mapping)
        h, w = grayscale_image.shape[:2] #to know if keypoints are in the image
        
        #get grayscale values at the selected points
        pixel_values = []
        for mapped_kp in keypoints:
            if mapped_kp is not None:
                x, y = mapped_kp
                if 0 <= x < w and 0 <= y < h:
                    pixel_values.append(grayscale_image[int(round(y)), int(round(x))]) # Ensure integer indices
                else:
                    pixel_values.append(float('nan'))
            else:
                pixel_values.append(float('nan'))
            
        # get temperatures by normalizing the value from grayscale to the temperature range
        temperatures = [normalize(value, 0, 255, min_temp, max_temp) if not math.isnan(value) else float('nan') for value in pixel_values]        
        
        return temperatures
    
        
######################################################################### MAIN ############################################################################

def process_line(line, i):
    print(i)
    # read the path, min temp and max temp
    process, path, mint, maxt, keypoints = read_file_line(line)
    newborn_id = path.split('/')[5]
    #print(newborn_id)
    
    if process == "True": ######## VIS Exists
        # -------- LOAD IMAGES -------- #
        base_path, ext = os.path.splitext(path)

        possible_vis_paths = [
            f"{base_path}_VIS{ext}",
            f"{base_path}.VIS{ext}"
        ]
        
        vis_path = possible_vis_paths[0] if os.path.exists(possible_vis_paths[0]) else possible_vis_paths[1]
        normal_img = Image.open(vis_path)
        thermal_img = Image.open(path)
        
        normal_width, normal_height = normal_img.size

        # -------- PROCESS -------- #
        # Convert keypoints to ints (they are strings)
        keypoints = [(int(x), int(y)) for x, y in keypoints]
        print(keypoints)
        mapped_keypoints = [map_normal_to_cropped(newborn_id, int(x), int(y), normal_width, normal_height, False) for x, y in keypoints]
        print(mapped_keypoints)

        temperatures = check_temperatures(path, newborn_id, maxt, mint, mapped_keypoints)
        print(temperatures)
        print("------------------------------------------------------------------------")

        # -------- SAVE VISUALIZATION -------- #
        '''
        # Draw on normal image
        normal_draw = ImageDraw.Draw(normal_img)
        for x, y in keypoints:
            normal_draw.ellipse([(x-5, y-5), (x+5, y+5)], outline="red", width=4)

        # Draw on thermal image
        thermal_draw = ImageDraw.Draw(thermal_img)
        for kp in mapped_keypoints:
            if kp[0] is not None:
                x, y = kp
                thermal_draw.ellipse([(x-5, y-5), (x+5, y+5)], outline="white", width=2)

        # Save to same folder
        print(f"./{i}normal_with_kp.jpg")
        normal_save_path = f"./{i}normal_with_kp.jpg"
        thermal_save_path = f"./{i}_thermal_with_kp.jpg"

        normal_img.save(normal_save_path)
        thermal_img.save(thermal_save_path)
        '''
        
        # -------- UPDATE CSV LINE -------- #
        temp_str = ",".join([f"{t:.2f}" for t in temperatures])
        new_line = line.strip() + "," + temp_str + "\n"
    else:
        new_line = line
        
    return (i, new_line)  # Return both index and processed line

def resolve_csv_paths(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    elif os.path.isdir(path):
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        return csv_files
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory")

def main(path):
    filepaths = resolve_csv_paths(path)
    
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        header = lines[0].strip()
        if not any(field in header for field in ["core_temp", "arm1_temp", "arm2_temp", "leg1_temp", "leg2_temp"]):
            header = header + ",core_temp,arm1_temp,arm2_temp,leg1_temp,leg2_temp"
        updated_lines = [header + "\n"]

        # Process lines in parallel (skip header)
        results = Parallel(n_jobs=-1) (
            delayed(process_line)(line, i) 
            for i, line in enumerate(lines[1:], 1))

        # Sort results by original index and extract just the lines
        results_sorted = [line for i, line in sorted(results, key=lambda x: x[0])]
        updated_lines.extend(results_sorted)
            
        # Now overwrite the file with the new content
        with open(filepath, 'w') as file:
            file.writelines(updated_lines)
            
        print(f"✅ Results saved to {filepath}")

if __name__ == "__main__":
    main("/home/cvmsct05/temperatures_max_min/40")
    
#"/home/cvmsct05/temperatures_max_min/29/20.08.24/20.08.24.csv"
