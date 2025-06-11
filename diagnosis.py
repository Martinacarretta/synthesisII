import os
import pandas as pd
import numpy as np
from pandas.errors import ParserError

"""
If process = False, then do not process

If no core temperature --> diagnosis = No core temperature

If no limbs to compute average --> only look for fever or hypothermia based on the core temperature

If core and limbs --> compute difference between core and limbs avg
    if core - avg_limbs > 2.0 --> diagnosis = Peripheral shutdown
"""


def diagnose(core, avg_limbs):
    if pd.isna(avg_limbs):
        if core >= 37.5:
            return "Fever"
        elif core < 35.0:
            return "Hypothermia"
        else:
            return "Normal"
    else:
        if core >= 37.5:
            if core - avg_limbs > 2.5:
                return "Fever and Peripheral shutdown"
            return "Fever"
        elif core < 35.0:
            if core - avg_limbs > 2.5:
                return "Hypothermia and Peripheral shutdown"
            return "Hypothermia"
        elif core - avg_limbs > 2.5:
            return "Peripheral shutdown"
        else:
            return "Normal"

def process_csv(input_csv_path, output_base="/home/cvmsct05/diagnosis/"):
    print(input_csv_path)

    # Read the CSV
    try:
        df = pd.read_csv(input_csv_path)
    except ParserError as e:
        print(f"❌ Skipping file due to ParserError: {input_csv_path}")
        print(f"   → {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error in file {input_csv_path}: {e}")
        return
    
    # Rename last 5 columns for clarity (if they’re unnamed)
    temp_cols = df.columns[-5:]
    df = df.rename(columns={temp_cols[4]: 'core', 
                            temp_cols[0]: 'left_hand', temp_cols[1]: 'right_hand', 
                            temp_cols[2]: 'left_foot', temp_cols[3]: 'right_foot'})
    
    output_rows = []

    # Read and process each row
    for _, row in df.iterrows():
        process_val = row["Process"]
        img_path = row["Image Path"]

        if not process_val:
            output_rows.append([False, img_path] + [""] * 8)
            continue

        # Handle missing core
        try:
            core = float(row["core"])
        except (ValueError, TypeError):
            core = np.nan

        if pd.isna(core): # If core temperature is missing
            output_rows.append([True, img_path] + [""] * 7 + ["No core temperature"])
            continue

        # Try to parse limb temperatures
        limbs = []
        for col in ["left_hand", "right_hand", "left_foot", "right_foot"]:
            try:
                val = float(row[col])
                limbs.append(val)
            except (ValueError, TypeError):
                continue  # Skip missing or invalid entries
        
        #get limb average
        limbs_avg = round(np.nanmean(limbs), 2) if limbs else np.nan
        
        #get difference of core and limbs
        core_minus_limbs = core - limbs_avg if not np.isnan(limbs_avg) else np.nan
        diagnosis = diagnose(core, limbs_avg)

        # Prepare the output row
        float_vals = [core] + limbs + [limbs_avg, core_minus_limbs]
        float_vals_rounded = [round(val, 2) if not np.isnan(val) else "" for val in float_vals]

        output_rows.append([True, img_path] + float_vals_rounded + [diagnosis])

    output_df = pd.DataFrame(output_rows, columns=[
        "Process", "Image Path",
        "core", "left_hand", "right_hand", "left_foot", "right_foot",
        "limbs_avg", "core_minus_limbs", "diagnosis"
    ])

    # Create output path
    rel_path = input_csv_path.replace("/home/cvmsct05/temperatures_max_min/", "")
    output_path = os.path.join(output_base, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the new CSV
    output_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def traverse_and_process(path):
    """
    Traverse the directory and process all CSV files found.
    If the path is a CSV file, it processes that file directly.
    If the path is a directory, it processes all CSV files within that directory.
    """
    if path.endswith(".csv"):
        process_csv(path)
    else:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    process_csv(full_path)

def main (input_path): # Main function used to call the traverse_and_process function easily from the if __name__ == "__main__" block
    traverse_and_process(input_path)
    
# Example usage
if __name__ == "__main__":
    main("/home/cvmsct05/temperatures_max_min/46")
    main("/home/cvmsct05/temperatures_max_min/55")
