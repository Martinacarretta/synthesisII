# synthesisII
# Sepsis Detection Using Thermal Imaging

## Context:
This project aims to detect early signs of sepsis in newborns using thermal imaging. Sepsis is a potentially life-threatening condition caused by the body's extreme response to an infection. Early detection is critical, especially in neonates, where symptoms are subtle and progress fast.

The approach leverages thermal cameras to monitor temperature gradients between the baby's core (torso) and limbs, since irregular differences in these regions can indicate abnormal thermoregulation — an early sepsis marker.

The goal is to develop a system that:

- Detects the baby and their limbs in thermal images

- Extracts reliable temperature readings at keypoints

- Calculates core-limb temperature differences

- Flags abnormal patterns for further clinical evaluation


## Pipeline Overview:
1. Extraction of maximum and minimum temperature of the thermal image (images with abnormally high minimum tempratures or abnormally low maximum temperatures are discarded)
    - [Temperature max and min](temperature_max_min.py)

2. Detection of the baby using Google's pose detection model
    - [keypoints](keypoints.py)

3. Keypoint Extraction (torso, left hand, right hand, left foot, right foot).
In this step, there are also heuristics to discard keypoints given the distance of shoulders, the distance from nose to feet, skin color...
    - [valid 5 keypoints](keypoints_valid.py)

4. Mapping of the keypoints from normal photograph to thermal (since the thermal is a cropped version of the normal). Conversion of the thermal image to grayscale (manual since the automatic conversion does not preserve the order of the temperatures). Extraction of the grayscale value of the keypoints. Conversion from grayscale value to thermal (using normalization with the min and max temperatures fo that photo) (discarding of keypints with abnormal temperatures)
    - [temperatures](temperatures.py)

5. CSV back-up to avoid incorrect modifications to the files in the detection phase. 
    - [save](save.py)

6. Temperature Difference Calculation
Compute ΔT = T_core - T_limb (being t_limb the average of the limbs)
Analyze asymmetry or abnormal values. Flag images with: Extreme ΔT values, fever, or hhypothermia
    - [diagnosis](diagnosis.py)

7. Creates the plots of the extracted temperatures. It has two different options; plots are created for each subfolder or for the entire baby folder. 
    - [plots](plots.py)

8. Smooths the plots created in the previous step using a rolling average of 45
    - [noise](noise.py)

9. Merges all the csv from the different baby subfolders into a single one
    - [joint](joint.py)

## Validation Overview:
1. [filter vis images](VALIDATION_SEPSIS/filter_vis_images.py): Filters images to include only the visual ones for annotating full days for Baby 47.

2. [keypoints](VALIDATION_SEPSIS/keypoints.ipynb): Used for annotating keypoints in thermal images.

3. [keypoints2](VALIDATION_SEPSIS/keypoints2.ipynb): Also used for annotating keypoints in visual images.

4. [plots_VIS](VALIDATION_SEPSIS/plots_VIS.py): Adapted code for visualizing time series plots of the visual images.

5. [temperatures_keypoints_complete](VALIDATION_SEPSIS/temperatures_keypoints_complete.py): Maps the temperatures of the annotated keypoints in the images.

6. [temperatures_keypoints_val](VALIDATION_SEPSIS/temperatures_keypoints_val.py): Attempts to map temperatures of the annotated keypoints in the thermal images, but only works when all five keypoints are annotated.

7. [visualize_annotations](VALIDATION_SEPSIS/visualize_annotations.py): Displays annotations from three different annotators to allow visual comparison.


## Tools and libraries:

- Python (NumPy, OpenCV, scikit-image, matplotlib)
- PyTorch / TensorFlow (for detection models)
- Pandas (data handling and ΔT logging)
- Jupyter Notebooks (prototyping)

## Useful directories:
[Save](save) is a backup of only the csv with the paths, min max temperatures and the keypoints

[diagnosis](diagnosis) is a folder of csv with the temperatures of the keypoints and the outputs given the assigned thresholds. 

[joint](joint_csv) is a folder with ony a csv per baby with all the paths, and temperatures detected. 


## Other used python scripts:
[check](others/check.py) checks if two csvs have the same content.

[draw](others/draw.py) draws different sized windows to see how to do the average window and the maximum window (used for temeprature extracting)

[find](others/find.py) finds the path of a file given the last part of the path and extension of the file 

[gif](others/gif.py) generates a gif of images with their corresponding temperature given a csv path

[move images](others/move_images.py) moves imgaes to another folder

[save](others/save.py) creates a backup of the provided directory

[SAM](others/sam.ipynb) (Segment Anything Model) trials

[U-net](others/unet.ipynb) trials

[yolo](others/yolo_pose.ipynb) pose detection model trials

[googlePose](others/googlePose.ipynb) pose detection model trials

[skinColor](others/skin_color.ipynb) skin color pixel classification

 
## Notes:
Data anonymization and ethical considerations are followed.

This is a proof-of-concept and not a diagnostic tool.

More work is needed to calibrate thermal readings to real-world temperatures and validate with clinical data.
