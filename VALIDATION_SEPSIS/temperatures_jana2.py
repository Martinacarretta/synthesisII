# FOR EXTRACTING VALIDATION TEMPERTATURES
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from joblib import Parallel, delayed
import os
from PIL import Image
import math


def read_file_line(line):
    parts = line.strip().split(',')
    
    if parts[1].strip().lower() == "image name":
        return None, None, None, None, None

    path = parts[1].strip()
    maxt = float(parts[2].strip())
    mint = float(parts[3].strip())
    keypoints = [
        (parts[4].strip(), parts[5].strip()),
        (parts[6].strip(), parts[7].strip()),
        (parts[8].strip(), parts[9].strip()),
        (parts[10].strip(), parts[11].strip()),
        (parts[12].strip(), parts[13].strip())
    ]
    
    return "True", path, mint, maxt, keypoints


def normalize(value, min_val, max_val, new_min, new_max):
    return new_min + (value - min_val) * (new_max - new_min) / (max_val - min_val)


def color_to_temp_to_grayscale(image, newborn_id, max_temp, min_temp):
    if newborn_id in ["29", "30", "31"]:
        temperature_colors = { 
            1.00: (255, 255, 255),
            0.95: (255, 255, 0),
            0.90: (255, 100, 0),
            0.80: (255, 0, 0),
            0.70: (255, 100, 100),
            0.60: (255, 200, 0),
            0.50: (0, 255, 0),
            0.40: (0, 128, 255),
            0.30: (0, 0, 255),
            0.20: (0, 0, 200),
            0.10: (75, 0, 130),
            0.00: (0, 0, 0)
        }
    else:
        temperature_colors = {
            0.00: (75, 0, 130),
            0.10: (0, 0, 200),
            0.20: (0, 0, 255),
            0.30: (0, 128, 255),
            0.40: (0, 255, 0),
            0.50: (173, 255, 47),
            0.60: (255, 255, 0),
            0.70: (255, 200, 0),
            0.75: (255, 165, 0),
            0.80: (255, 100, 0),
            0.90: (255, 0, 0),
            0.95: (255, 100, 100),
            1.00: (255, 255, 255)
        }

    normalized_temps = np.array(list(temperature_colors.keys()))
    colors = np.array(list(temperature_colors.values()), dtype=np.uint8)
    
    colors_lab = rgb2lab(colors.reshape(1, -1, 3)).reshape(-1, 3)
    image_reshaped = image.reshape(-1, 3).astype(np.uint8)
    image_lab = rgb2lab(image_reshaped.reshape(1, -1, 3)).reshape(-1, 3)
    
    num_chunks = 8
    chunks = np.array_split(image_lab, num_chunks)
    
    def process_chunk(chunk):
        chunk_expanded = chunk[:, np.newaxis, :]
        colors_lab_expanded = colors_lab[np.newaxis, :, :]
        distances = deltaE_ciede2000(chunk_expanded, colors_lab_expanded)
        closest_indices = np.argmin(distances, axis=1)
        return normalized_temps[closest_indices]
    
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk) for chunk in chunks)
    temperature_map = np.concatenate(results).reshape(image.shape[:2])
    actual_temp_values = temperature_map * (max_temp - min_temp) + min_temp
    grayscale_image = np.clip((actual_temp_values - min_temp) / (max_temp - min_temp) * 255, 0, 255).astype(np.uint8)
    
    return grayscale_image



def map_normal_to_cropped(baby_ID, x, y, normal_width, normal_height, normalized=True):
    if x == None or y == None:
        return None, None
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

def grayscale_image_creation (path, newborn_id, max_temp, min_temp):
    try:
        pil_image = Image.open(path) #thermal
        image = pil_image.convert("RGB")
        # Convert PIL Image to numpy array
        image = np.array(image)
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        return None
    
    grayscale_image = color_to_temp_to_grayscale(image, newborn_id, max_temp, min_temp) # RGB --> grayscale (with temperature mapping)

    return grayscale_image

def get_temperature_at_point (coord, grayscale_image, min_temp, max_temp):
    h, w = grayscale_image.shape[:2] #to know if keypoints are in the image
    if coord is None or None in coord:
        return float('nan')

    x, y = coord
    if 0 <= x < w and 0 <= y < h:
        value = grayscale_image[int(round(y)), int(round(x))]
        return normalize(value, 0, 255, min_temp, max_temp)
    else:
        return float('nan')

def check_temperatures(path, newborn_id, max_temp, min_temp, keypoints):
    try:
        pil_image = Image.open(path)
        image = np.array(pil_image.convert("RGB"))
    except Exception as e:
        print(f"Error loading image {path}: {str(e)}")
        return [float('nan')] * len(keypoints)
    
    grayscale_image = color_to_temp_to_grayscale(image, newborn_id, max_temp, min_temp)
    h, w = grayscale_image.shape[:2]
    
    pixel_values = []
    for kp in keypoints:
        if kp is None or None in kp:
            pixel_values.append(float('nan'))
            continue
        x, y = kp
        if x is not None and y is not None and 0 <= int(x) < w and 0 <= int(y) < h:
            pixel_values.append(grayscale_image[int(round(float(y))), int(round(float(x)))])
        else:
            pixel_values.append(float('nan'))

    temperatures = [normalize(v, 0, 255, min_temp, max_temp) if not math.isnan(v) else float('nan') for v in pixel_values]
    return temperatures


def process_line(line, i):
    print(i)
    try:
        process, path, mint, maxt, keypoints = read_file_line(line)
        
        if process == "True":
            newborn_id = path.split('/')[5]
            base_path, ext = os.path.splitext(path)
            possible_vis_paths = [f"{base_path}_VIS{ext}", f"{base_path}.VIS{ext}"]
            vis_path = possible_vis_paths[0] if os.path.exists(possible_vis_paths[0]) else possible_vis_paths[1]
            
            if not os.path.exists(path) or not os.path.exists(vis_path):
                print(f"?? Skipping line {i} due to missing image(s): {path}, {vis_path}")
                raise FileNotFoundError

            keypoints = [(int(x), int(y)) if x.strip() and y.strip() else (None, None) for x, y in keypoints]
            # Abrir imagen VIS para conocer tamaño original
            vis_image = Image.open(vis_path)
            w_vis, h_vis = vis_image.size

            # Adaptar keypoints de VIS a imagen térmica
            mapped_keypoints = [
                map_normal_to_cropped(newborn_id, x, y, w_vis, h_vis, normalized=False) if x is not None and y is not None else (None, None)
                for x, y in keypoints
            ]

            temperatures = check_temperatures(path, newborn_id, maxt, mint, mapped_keypoints)

            print("------------------------------------------------------------------------")
            temp_str = ",".join([f"{t:.2f}" for t in temperatures])
            new_line = line.strip() + "," + temp_str + "\n"
        else:
            new_line = line

    except Exception as e:
        print(f"? Error processing line {i}: {str(e)}")
        new_line = line.strip() + "," + ",".join(["nan"] * 5) + "\n"

    return (i, new_line)


def resolve_csv_paths(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        return [path]
    elif os.path.isdir(path):
        return [os.path.join(root, file) 
                for root, _, files in os.walk(path) 
                for file in files if file.endswith(".csv")]
    else:
        raise ValueError(f"Path '{path}' is not a valid file or directory")


def main(path):
    filepaths = resolve_csv_paths(path)
    required_fields = ["torso_temp", "left_hand_temp", "right_hand_temp", "left_foot_temp", "right_foot_temp"]

    for filepath in filepaths:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        header = lines[0].strip()
        if all(field in header for field in required_fields):
            print(f"?? Skipping {filepath} (already processed)")
            continue
        
        header += "," + ",".join(f for f in required_fields if f not in header)
        updated_lines = [header + "\n"]

        results = Parallel(n_jobs=-1)(
            delayed(process_line)(line, i)
            for i, line in enumerate(lines[1:], 1)
        )

        results_sorted = [line for _, line in sorted(results, key=lambda x: x[0])]
        updated_lines.extend(results_sorted)

        dir_name = os.path.dirname(filepath)
        base_name = os.path.basename(filepath)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_with_temps{ext}")

        with open(output_path, 'w') as file:
            file.writelines(updated_lines)

        print(f"? Results saved to {output_path}")


if __name__ == "__main__":
    main("/data/uabcvmsc/cvmsct05/ValidationManualCsvs/prova_vis/")  # Replace with your path
