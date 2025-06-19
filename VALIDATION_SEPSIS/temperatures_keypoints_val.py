# -*- coding: utf-8 -*- 
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from joblib import Parallel, delayed
import os
from PIL import Image
import math


def read_file_line(line):
    parts = line.strip().split(',')
    
    if parts[0].strip().lower() == "image name":
        return None, None, None, None, None

    path = parts[0].strip()
    maxt = float(parts[1].strip())
    mint = float(parts[2].strip())
    keypoints = [
        (parts[3].strip(), parts[4].strip()),
        (parts[5].strip(), parts[6].strip()),
        (parts[7].strip(), parts[8].strip()),
        (parts[9].strip(), parts[10].strip()),
        (parts[11].strip(), parts[12].strip())
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
            
            keypoints = [(int(x), int(y)) if x not in (None, 'None') and y not in (None, 'None') else (None, None) for x, y in keypoints]
            temperatures = check_temperatures(path, newborn_id, maxt, mint, keypoints)
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
    import sys
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <path_to_csv_or_folder>")
    else:
        main(sys.argv[1])
