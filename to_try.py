#### GIVES RANDOM COORDINATES. IT'S JUST A TRIAL TO SEE I CAN CORRECTLY EXTRACT THE TEMPERATURES FROM THE KEYPOINTS THAT WOULD BE GIVEN TO ME IN THE CSV

# Coordinates you want to add (core, arm1, arm2, leg1, leg2)
#keypoints = [(301, 45), (116, 71), (344, 179), (59, 448), (45, 609)]
keypoints = [(287, 53), (535, 92), (277, 159), (551, 235), (317, 402)]

# Input and output file paths
input_file = "/home/cvmsct05/trial.csv"
output_file = "/home/cvmsct05/trial2.csv"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    lines = infile.readlines()
    
    # Keep the header unchanged
    header = lines[0].strip()
    outfile.write(header + ",core_x,core_y,arm1_x,arm1_y,arm2_x,arm2_y,leg1_x,leg1_y,leg2_x,leg2_y\n")
    
    # Add the same keypoints to each data row
    for line in lines[1:]:
        if not line.strip():
            continue
        line = line.strip()
        flat_coords = [str(num) for point in keypoints for num in point]
        outfile.write(line + "," + ",".join(flat_coords) + "\n")
