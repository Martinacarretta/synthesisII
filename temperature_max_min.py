import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import re
from skimage.color import deltaE_ciede2000, rgb2lab
import csv
import glob
import os

# Script to process images and extract temperature minimum and maximum using Tesseract OCR
# important to note this is the only script that works with opencv, the rest use PIL, so it's okay to work with vertical and horizontal images

def process_region(region):
    """
    This function takes a region of an image as input, preprocesses it, and uses Tesseract OCR to extract text from the region.
    """
    # Preprocess the image (convert to grayscale and apply thresholding)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV) #thresh is the image

    # Use Tesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    numbers = re.findall(r"\d+\.?\d*", text)
    
    for i in range(len(numbers)): # add decimal point if there isn't
        if len(numbers[i]) == 3:
            numbers[i] = numbers[i][:2] + "." + numbers[i][2]

    return numbers #returns the numbers just like that 

def all_images (path, plot=True):
    """
    Given a path to an image, it will create the windows for where the maximum and minimum temperatures are located, and return the maximum and minimum temperatures.
    """
    # Iterate through each image in the folder
    print(f"Processing: {path}")

    image = cv2.imread(path)
    if image is None:
        print(f"Error loading image: {path}")
        return None, None
    
    # GET WINDOW COORDINATES
    if image.shape[1] > image.shape[0]: #landscape
        u = [15, 139, 70, 33]
        l = [15, 392, 70, 33]
    else:
        u = [15, 139, 70, 33]
        l = [15, 455, 70, 33]
    
    ux, uy, uw, uh = u  # Upper region
    lx, ly, lw, lh = l  # Lower region
    
    square_region_upper = image[uy:uy+uh, ux:ux+uw] #max temp square
    square_region_lower = image[ly:ly+lh, lx:lx+lw] #min temp square
    
    # call the function to get the min and max temperatures
    numbers = process_region(square_region_upper)
    numbers2 = process_region(square_region_lower)

    # return the maximum and minimum temperatures
    try:
        upper = float(max(numbers)) if numbers else None
        lower = float(min(numbers2)) if numbers2 else None
        return upper, lower
    except (ValueError, TypeError):
        return None, None

    
def process_folder (folder_path):
    """
    This function processes all images in a given folder, extracts temperature data, and saves the results to a CSV file.
    """
    # Only consider image files that do not contain "VIS" in their name since those are not temperature images
    image_paths = sorted([f for f in glob.glob(os.path.join(folder_path, "*")) 
                         if f.lower().endswith(('.jpeg', '.jpg', '.png')) and "VIS" not in f])

    # Extract parts of the path for folder and filename
    path_parts = folder_path.strip("/").split("/")
    
    folder_name = path_parts[-2]  # baby
    file_name = path_parts[-1]    # day

    # Define the output directory and CSV file name
    output_dir = f"temperatures_max_min/{folder_name}/{file_name}"
    csv_filename = f"{output_dir}/{file_name}.csv"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the CSV file in write mode
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(["Process", "Image Path", "Max Temp", "Min Temp"])
        
        # Process each image
        i=0 # this is just used to see the progress in the terminal
        for path in image_paths:
            print(i) # this is just used to see the progress in the terminal
            i+=1
            max_t, min_t = all_images(path) # call function
            
            if max_t is not None: # Haven't had problems with min_t being None
                if max_t < 33 or max_t > 50: # Threshold for max temperature
                    process = False # set parameter to false if the max temperature is not in the range
                else:
                    process = True
            else:
                process = False
                
            # Write data to CSV
            writer.writerow([process, path, max_t, min_t])

    print(f"✅ Results saved to {csv_filename}")
    
def main(root_path):
    """
    Traverse the path and see if it's a day folder, a baby folder or a root directory.
    If it's a day folder, process it. If it's a baby folder, process all day folders inside it.
    If it's the root directory, process all day folders inside it.
    """
    def is_day_folder(path):
        return (
            os.path.isdir(path)
            and any(f.lower().endswith(('.jpeg', '.jpg', '.png')) for f in os.listdir(path))
            and all(not os.path.isdir(os.path.join(path, f)) for f in os.listdir(path))
        )


    # Case 1: The root itself is a day folder
    if os.path.isdir(root_path) and is_day_folder(root_path):
        print(f"📁 Processing single day folder: {root_path}")
        process_folder(root_path)
        return

    # Case 2: The root is a baby folder or root directory
    for baby_id in os.listdir(root_path):
        baby_path = os.path.join(root_path, baby_id)
        if not os.path.isdir(baby_path):
            continue

        # If this baby folder is actually a day folder
        if is_day_folder(baby_path):
            print(f"📁 Processing day folder: {baby_path}")
            process_folder(baby_path)
            continue

        # Otherwise, it should contain days
        for day in os.listdir(baby_path):
            day_path = os.path.join(baby_path, day)
            if not os.path.isdir(day_path):
                continue
            if is_day_folder(day_path):
                print(f"📁 Processing day folder: {day_path}")
                process_folder(day_path)

    
if __name__ == "__main__":
    main("/data/uabcvmsc/shared/newborn/49")     # Default path when run directlyk