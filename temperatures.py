import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from joblib import Parallel, delayed
import cv2
import matplotlib.pyplot as plt
import random
import os
import glob
from PIL import Image
import pytesseract
import re
import math


######################## STEP 1: return path of image, min temp, max temp ########################

def read_file_line (line):
    # Split the line by commas and strip whitespace
    parts = line.strip().split(',')
    
    # Extract the path and temperature values
    path = parts[0].strip()
    mint = float(parts[1].strip())
    maxt = float(parts[2].strip())
    
    #extract keypoints we want the temp of
    keypoints = [(parts[3].strip(), parts[4].strip()), #core
                 (parts[5].strip(), parts[6].strip()), #arm1
                 (parts[7].strip(), parts[8].strip()), #arm2
                 (parts[9].strip(), parts[10].strip()), #leg1
                 (parts[11].strip(), parts[12].strip()) #leg2
                ]
    
    return path, mint, maxt, keypoints
    

######################## STEP 2: convert to grayscale ########################

def normalize(value, min_val, max_val, new_min, new_max):
    return new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val)


def color_to_temp_to_grayscale(image, max_temp, min_temp):
    # Define a fine-grained normalized temperature-to-color mapping TO PRESERVE THE ORDER OF THE TEMPERATURE SCALE. 
    # DICTIONARY GOES FROM 0 TO 1 SO THAT WE CAN USE THE MAX AND MIN TEMPERATURE VALUES TO NORMALIZE THE TEMPERATURES FOR EACH PICTURE ACCORDINGLY
    if image.shape[1] > image.shape[0]: #landscape
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
    else:
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


######################## STEP 3: check the temperatures of the keypoints ########################

def check_temperatures(path, max_temp, min_temp, keypoints):
        image_normal = cv2.imread(path)
        if image_normal is None:
            print(f"Error loading image: {path}")
            return
        
        image = cv2.cvtColor(image_normal, cv2.COLOR_BGR2RGB)
        
        grayscale_image = color_to_temp_to_grayscale(image, max_temp, min_temp) # RGB --> grayscale (with temperature mapping)
        h, w = grayscale_image.shape[:2] #to know if keypoints are in the image
        
        #get grayscale values at the selected points
        pixel_values = []
        for x, y in keypoints:
            if 0 <= x < w and 0 <= y < h:
                pixel_values.append(grayscale_image[y, x])
            else:
                pixel_values.append(float('nan'))
            
        # get temperatures by normalizing the value from grayscale to the temperature range
        temperatures = [normalize(value, 0, 255, min_temp, max_temp) if not math.isnan(value) else float('nan') for value in pixel_values]        
        
        
        # Print #################### see if necessary ...
        """
        print("coordinates - grayscale value - temperatures:")
        for (x, y), pixel, temp in temperatures:
            print(f"({x}, {y}) - Grayscale value: {pixel}, Temperature: {temp:.1f}")
        """
        
        return temperatures
    
        
filepath = "/home/cvmsct05/temperatures_max_min/29/14.08.24-15.08.24.csv"

updated_lines = []

with open(filepath, 'r') as file:
    for line in file:
        # read the path, min temp and max temp
        path, mint, maxt, keypoints = read_file_line(line)
        
        # Convert keypoints to ints (they are strings)
        keypoints = [(int(x), int(y)) for x, y in keypoints]

        temperatures = check_temperatures(path, maxt, mint, keypoints)
        
        # Format temperatures as strings
        temp_str = ",".join([f"{t:.2f}" for t in temperatures])
        
        # Append to line and save
        new_line = line.strip() + "," + temp_str + "\n"
        updated_lines.append(new_line)
        
# Now overwrite the file with the new content
with open(filepath, 'w') as file:
    file.writelines(updated_lines)
    
