# input_folder = r"/home/cvmsct05/temperatures_max_min/29/14.08.24-15.08.24/valid_detection"
# output_folder = r"/home/cvmsct05/temperatures_max_min/29/14.08.24-15.08.24"
# import os
# import shutil
# #take all the files in the input folder and move them to output folder
# for root, dirs, files in os.walk(input_folder):
#     for file in files:
#         input_file = os.path.join(root, file)
#         output_file = os.path.join(output_folder, file)
#         shutil.move(input_file, output_file)
#         print(f"Moved {input_file} to {output_file}")

from pathlib import Path

baby_folder = Path("/home/cvmsct05/temperatures_max_min/44")
subfolders = [f for f in baby_folder.iterdir() if f.is_dir()]

aaa = "temperatures_max_min/44/09.11.24"
for folder in subfolders:
    print(folder)
    if folder == aaa:
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        # out = Path(folder)
        # print(out)

        # csv_files = list(folder.rglob("*.csv"))

        # csv_file = csv_files[0]
        # output = out / "valid_detection"
        # output_valid_points = out / "valid_points"

    



# out = Path("temperatures_max_min/33/8.9.24")









