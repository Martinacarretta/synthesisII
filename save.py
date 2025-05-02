import os
import shutil

############### SCRIPT TO SAVE ALL THE CSV WE HAVE IN THE TEMPERATURES_MAX_MIN FOLDER JUST IN CASE ###############


def copy_csv_files_with_structure(src_path, dest_path):
    # Walk through the source directory
    for root, dirs, files in os.walk(src_path):
        # Filter CSV files
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        
        # Determine relative path and destination folder
        relative_path = os.path.relpath(root, src_path)
        dest_folder = os.path.join(dest_path, relative_path)

        # Create destination subfolder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)

        # Copy each CSV file
        for file in csv_files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_folder, file)
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {src_file} -> {dest_file}")

# Example usage:
source_directory = "/home/cvmsct05/temperatures_max_min"
destination_directory = "/home/cvmsct05/save"
copy_csv_files_with_structure(source_directory, destination_directory)

def delete_empty_folders(root_dir): #delete the valid points and valid detection folders
    root_dir = os.path.abspath(root_dir)

    # Traverse from bottom to top so that parent folders are checked after children
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
            except OSError as e:
                print(f"Failed to delete {dirpath}: {e}")
                
delete_empty_folders(destination_directory)