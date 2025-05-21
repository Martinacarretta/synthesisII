import os


### JUST TO FIND A PATH BASED ON THE PHOTO NAME

def find_image(start_path, target_filename="HM20241222154347.jpeg"):
    for root, dirs, files in os.walk(start_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

# Example usage:
if __name__ == "__main__":
    folder_path = "/data/uabcvmsc/shared/newborn"  # Change this to your starting folder
    result = find_image(folder_path)

    if result:
        print(f"Image found: {result}")
    else:
        print("Image not found.")
