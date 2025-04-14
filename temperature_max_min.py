import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import re
import random
from skimage.color import deltaE_ciede2000, rgb2lab
import csv
import glob
import os


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
    # Iterate through each image in the folder
    print(f"Processing: {path}")

    image = cv2.imread(path)
    if image is None:
        print(f"Error loading image: {path}")
        return
      
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
    
    numbers = process_region(square_region_upper)
    numbers2 = process_region(square_region_lower)

    return float(max(numbers)), float(min(numbers2))

    
def main():
    folder_path = "/data/uabcvmsc/shared/newborn/29/15.08.24"
    
    image_paths = sorted([f for f in glob.glob(os.path.join(folder_path, "*")) if "VIS" not in f])
    
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
        i=0
        for path in image_paths:
            print(i)
            i+=1
            max_t, min_t = all_images(path)
            
            if max_t < 33:
                process = False
            else:
                process = True
            # Write data to CSV
            writer.writerow([process, path, max_t, min_t])

    print(f"✅ Results saved to {csv_filename}")

    
if __name__ == "__main__":
    main()

