import os
import os.path as osp
import glob
import base64
import json
import argparse
import numpy as np
from labelme.label_file import LabelFile
import labelme.utils  # Import labelme.utils to fix the NameError
import imgviz  # For visualization (optional)

def create_json_for_image(image_path, output_json_path):
    """Create a JSON file for an image with no bounding boxes."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": osp.basename(image_path),
        "imageData": image_data,
    }

    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)

def convert(size, box):
    """Convert bounding box coordinates to YOLO format."""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_labelme_to_yolo(input_dir, output_dir, labels_file, noviz=False):
    """Convert LabelMe annotations to YOLO format while retaining folder structure."""
    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        return

    # Create the required folders
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "JPEGImages"))
    os.makedirs(osp.join(output_dir, "Annotations"))
    if not noviz:
        os.makedirs(osp.join(output_dir, "AnnotationsVisualization"))
    print("Creating dataset:", output_dir)

    # Read class names from the labels file
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(labels_file).readlines()):
        class_name = line.strip()
        if class_name == "_background_":  # Skip the background class
            continue
        class_id = len(class_names)  # Assign IDs starting from 0
        class_name_to_id[class_name] = class_id
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)

    # Save class names to a file
    out_class_names_file = osp.join(output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # Process images in the input directory
    for image_path in glob.glob(osp.join(input_dir, "*.jpg")) + glob.glob(osp.join(input_dir, "*.jpeg")):
        base = osp.splitext(osp.basename(image_path))[0]
        json_path = osp.join(input_dir, base + ".json")

        # If no JSON file exists, create one
        if not osp.exists(json_path):
            print(f"No JSON file found for {image_path}. Creating one...")
            create_json_for_image(image_path, json_path)

        # Process the JSON file
        try:
            label_file = LabelFile(filename=json_path)
        except Exception as e:
            print(f"Skipping {json_path} due to error: {e}")
            continue

        # Save the image to JPEGImages folder
        out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        # Save YOLO annotations to Annotations folder
        out_txt_file = osp.join(output_dir, "Annotations", base + ".txt")
        height, width = img.shape[:2]

        bboxes = []
        labels = []
        with open(out_txt_file, "w") as f:
            for shape in label_file.shapes:
                if shape["shape_type"] != "rectangle":
                    print(f"Skipping shape: label={shape['label']}, shape_type={shape['shape_type']}")
                    continue

                class_name = shape["label"]
                if class_name not in class_names:
                    print(f"Skipping unknown class: {class_name}")
                    continue

                # Assign class IDs starting from 0
                class_id = class_names.index(class_name)
                (xmin, ymin), (xmax, ymax) = shape["points"]
                xmin, xmax = sorted([xmin, xmax])
                ymin, ymax = sorted([ymin, ymax])

                bbox = convert((width, height), (xmin, xmax, ymin, ymax))
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

                bboxes.append([ymin, xmin, ymax, xmax])
                labels.append(class_id)

        # Save visualization (optional)
        if not noviz:
            out_viz_file = osp.join(output_dir, "AnnotationsVisualization", base + ".jpg")
            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                bboxes=bboxes,
                captions=captions,
                font_size=15,
            )
            imgviz.io.imsave(out_viz_file, viz)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    convert_labelme_to_yolo(args.input_dir, args.output_dir, args.labels, args.noviz)

if __name__ == "__main__":
    main()