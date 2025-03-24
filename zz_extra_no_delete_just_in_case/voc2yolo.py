import glob
import os
import xml.etree.ElementTree as ET

# Define the directories to process
annotations_dir = '/home/martina/codi2/3year/synthesisII/bbox_detection/prova_dataset/Annotations'
images_dir = '/home/martina/codi2/3year/synthesisII/bbox_detection/prova_dataset/JPEGImages'

# Path to the text file containing class names (one class per line)
classes_file = '/home/martina/codi2/3year/synthesisII/bbox_detection/labels.txt'

def read_classes(file_path):
    """
    Read class names from a text file.
    Each line in the file represents a class name.
    """
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file.readlines() if line.strip()]
    return classes

def getImagesInDir(dir_path):
    """
    Get a list of all image file paths in the specified directory.
    Supports common image extensions: .jpg, .jpeg, .png.
    """
    image_list = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images = glob.glob(os.path.join(dir_path, ext))
        image_list.extend(images)
    return image_list

def convert(size, box):
    """
    Convert Pascal VOC bounding box coordinates (xmin, xmax, ymin, ymax)
    to YOLO format (center_x, center_y, width, height), normalized to [0, 1].
    """
    dw = 1. / size[0]  # Normalization factor for width
    dh = 1. / size[1]  # Normalization factor for height
    x = (box[0] + box[1]) / 2.0 - 1  # Calculate center x
    y = (box[2] + box[3]) / 2.0 - 1  # Calculate center y
    w = box[1] - box[0]  # Calculate width
    h = box[3] - box[2]  # Calculate height
    x = x * dw  # Normalize center x
    w = w * dw  # Normalize width
    y = y * dh  # Normalize center y
    h = h * dh  # Normalize height
    return (x, y, w, h)

def convert_annotation(dir_path, image_path, classes):
    """
    Convert a Pascal VOC annotation file (XML) to YOLO format (TXT).
    The YOLO annotation file (.txt) will be saved in the same directory as the XML file.
    If no objects are found, an empty .txt file will be created.
    """
    # Extract the base filename (without extension) from the image path
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    # Define input (XML) and output (TXT) file paths
    in_file = os.path.join(dir_path, basename_no_ext + '.xml')
    out_file = os.path.join(dir_path, basename_no_ext + '.txt')  # Save .txt in the same directory as XML

    # Skip if the XML file doesn't exist
    if not os.path.exists(in_file):
        print(f"Warning: {in_file} does not exist. Skipping.")
        return

    # Parse the XML file
    tree = ET.parse(in_file)
    root = tree.getroot()

    # Get image dimensions from the XML file
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # Open the output file to write YOLO annotations
    with open(out_file, 'w') as out_file:
        # Flag to check if any objects were found
        objects_found = False

        # Iterate over all objects in the XML file
        for obj in root.iter('object'):
            # Handle missing or invalid 'difficult' field
            difficult = obj.find('difficult')
            if difficult is not None and difficult.text is not None and difficult.text.strip().isdigit():
                difficult = int(difficult.text)
            else:
                difficult = 0  # Default to 0 if 'difficult' is missing or invalid

            cls = obj.find('name').text  # Get the class name

            # Skip if the class is not in the list or is marked as difficult
            if cls not in classes or difficult == 1:
                continue

            # Get the class ID (index in the classes list)
            cls_id = classes.index(cls)

            # Extract bounding box coordinates from the XML
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

            # Convert bounding box to YOLO format
            bb = convert((w, h), b)

            # Write the YOLO annotation to the output file
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            # Set the flag to True if at least one object is found
            objects_found = True

        # If no objects were found, create an empty .txt file
        if not objects_found:
            print(f"No objects found in {in_file}. Creating an empty .txt file.")
            # The file is already opened in 'w' mode, so it will be empty

def main():
    """
    Main function to process all directories and convert annotations.
    """
    # Read classes from the text file
    classes = read_classes(classes_file)
    print(f"Classes loaded: {classes}")

    # Get all image paths in the images directory
    image_paths = getImagesInDir(images_dir)

    # Process each image and convert its corresponding annotation
    for image_path in image_paths:
        convert_annotation(annotations_dir, image_path, classes)

    print(f"Finished processing: {annotations_dir}")

if __name__ == "__main__":
    # Run the main function
    main()