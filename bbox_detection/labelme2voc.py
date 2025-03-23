import os
import os.path as osp
import glob
import base64
import json
import argparse

import imgviz
import labelme
import numpy as np
from labelme.label_file import LabelFile

try:
    import lxml.builder
    import lxml.etree
except ImportError:
    print("Please install lxml:\n\n    pip install lxml\n")
    sys.exit(1)


def create_json_for_image(image_path, output_json_path):
    """Create a JSON file for an image with no bounding boxes."""
    # Encode the image in base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Create JSON structure
    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": osp.basename(image_path),
        "imageData": image_data,
    }

    # Save JSON to a file
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input annotated directory")
    parser.add_argument("output_dir", help="output dataset directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "Annotations"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "AnnotationsVisualization"))
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # Process images in the input directory
    for image_path in glob.glob(osp.join(args.input_dir, "*.jpg")) + glob.glob(
        osp.join(args.input_dir, "*.jpeg")
    ):
        base = osp.splitext(osp.basename(image_path))[0]
        json_path = osp.join(args.input_dir, base + ".json")

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

        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_xml_file = osp.join(args.output_dir, "Annotations", base + ".xml")
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir, "AnnotationsVisualization", base + ".jpg"
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)

        maker = lxml.builder.ElementMaker()
        xml = maker.annotation(
            maker.folder(),
            maker.filename(base + ".jpg"),
            maker.database(),  # e.g., The VOC2007 Database
            maker.annotation(),  # e.g., Pascal VOC2007
            maker.image(),  # e.g., flickr
            maker.size(
                maker.height(str(img.shape[0])),
                maker.width(str(img.shape[1])),
                maker.depth(str(img.shape[2])),
            ),
            maker.segmented(),
        )

        bboxes = []
        labels = []
        for shape in label_file.shapes:
            if shape["shape_type"] != "rectangle":
                print(
                    "Skipping shape: label={label}, shape_type={shape_type}".format(
                        **shape
                    )
                )
                continue

            class_name = shape["label"]
            class_id = class_names.index(class_name)

            (xmin, ymin), (xmax, ymax) = shape["points"]
            # swap if min is larger than max.
            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_id)

            xml.append(
                maker.object(
                    maker.name(shape["label"]),
                    maker.pose(),
                    maker.truncated(),
                    maker.difficult(),
                    maker.bndbox(
                        maker.xmin(str(xmin)),
                        maker.ymin(str(ymin)),
                        maker.xmax(str(xmax)),
                        maker.ymax(str(ymax)),
                    ),
                )
            )

        if not args.noviz:
            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                bboxes=bboxes,
                captions=captions,
                font_size=15,
            )
            imgviz.io.imsave(out_viz_file, viz)

        with open(out_xml_file, "wb") as f:
            f.write(lxml.etree.tostring(xml, pretty_print=True))


if __name__ == "__main__":
    main()