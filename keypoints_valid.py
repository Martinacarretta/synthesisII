import math
import csv

POSE_LANDMARKS = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}

POSE_LANDMARKS_BY_NAME = {v: k for k, v in POSE_LANDMARKS.items()}


def compute_distance(p1, p2):
    """
    Compute Euclidean distance between two points.
    Each point is a tuple (x, y).
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def parse_keypoints(row):
    keypoints = {}
    data = row
    for i in range(33):
        base_idx = 4 + i * 5
        if base_idx + 4 >= len(data):
            continue  # skip if incomplete data
        keypoints[POSE_LANDMARKS[i]] = {
            'x': float(data[base_idx]),
            'y': float(data[base_idx+1]),
            'z': float(data[base_idx+2]),
            'visibility': float(data[base_idx+3]),
            'presence': float(data[base_idx+4])
        }
    return keypoints

def narrow_torso(keypoints):
    left_shoulder_keypoints = keypoints['left_shoulder']
    right_shoulder_keypoints = keypoints['right_shoulder']

    ls_x = left_shoulder_keypoints['x']
    ls_y = left_shoulder_keypoints['y']    

    rs_x = right_shoulder_keypoints['x']
    rs_y = right_shoulder_keypoints['y']

    #calculate distance between left and right shoulder
    shoulder_distance = compute_distance((ls_x, ls_y), (rs_x, rs_y))
    if shoulder_distance < 300:
        return True

    return False

def total_fail(keypoints):
    nose_keypoints = keypoints['nose']
    left_heel_keypoints = keypoints['left_heel']

    n_x = nose_keypoints['x']
    n_y = nose_keypoints['y']

    lf_x = left_heel_keypoints['x']
    lf_y = left_heel_keypoints['y']

    distance = compute_distance((n_x, n_y), (lf_x, lf_y))
    if distance > 500 and distance < 600:
        return True
    return False



csv_file = r"t_save/29/14.08.24-15.08.24/14.08.24-15.08.24.csv"
rows = []
with open(csv_file, 'r', newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows = [row for row in reader]

process = headers.index("Process")

for row in rows:
    if row[0]=='True':
        keypoints = parse_keypoints(row)
        process_value = row[process]
        # if narrow_torso(keypoints):
        #     print('narrow torso')
        #     # process_value = 'False'
        if total_fail(keypoints):
            print(row[1])


