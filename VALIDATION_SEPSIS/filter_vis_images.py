import os
import shutil

# Cluster paths (no raw strings needed in Linux)
source_folder = "/data/uabcvmsc/shared/newborn/47/01.12.24 (11)/"
destination_folder = "/data/uabcvmsc/cvmsct05/ValidationManualCsvs/47/01.12.24 (11)_VIS/"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Filter and copy .VIS.jpeg files
for filename in os.listdir(source_folder):
    if ".VIS.jpeg" in filename:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(destination_folder, filename)
        shutil.copy2(src, dst)

print("✅ .VIS.jpeg images successfully copied to target folder.")



