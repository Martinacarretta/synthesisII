import os
import pandas as pd
import re

# This script merges multiple CSV files from a specified folder, ensuring they contain specific columns.
# It also sorts the merged data based on timestamps extracted from filenames in the "Image Path" column.

#This script is used to join all csv files of the same baby into one. 
# Used for the plots (). 

# Define the columns you want to keep
COLUMNS_TO_KEEP = [
    "Process", "Image Path", 
    "left_hand_temp", "right_hand_temp", 
    "left_foot_temp", "right_foot_temp",
    "torso_updated_temp"
]

def extract_filename_timestamp(path):
    """Extract a sortable timestamp from filename like HM20240827124854.jpeg."""
    try:
        # Clean up whitespace or weird characters
        filename = os.path.basename(str(path).strip())
        match = re.match(r"HM(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\.jpeg", filename)
        if match:
            # Convert to an actual datetime object (cleaner for sorting)
            from datetime import datetime
            return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except Exception as e:
        pass
    return None  # So it gets sorted to the bottom

    
    
def join_all_csvs(folder_path):
    all_dfs = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)
                    if not all(col in df.columns for col in COLUMNS_TO_KEEP):
                        print(f"❌ CSV doesn't have the indicated columns: {full_path}")
                        return  # Stop the whole function

                    filtered = df[[col for col in COLUMNS_TO_KEEP if col in df.columns]]
                    all_dfs.append(filtered)
                except Exception as e:
                    print(f"Skipping {full_path}: {e}")

    if not all_dfs:
        print("No valid CSVs found.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by timestamp from Path column
    if "Image Path" in merged_df.columns:
        merged_df["timestamp"] = merged_df["Image Path"].apply(extract_filename_timestamp)

        # Drop rows where timestamp couldn't be extracted, or sort with NaNs at the end
        merged_df = merged_df.sort_values(by="timestamp", na_position="last").drop(columns="timestamp")

    # Extract last folder name from input path to name the output file
    last_folder = os.path.basename(os.path.normpath(folder_path))
    os.makedirs("joint_csv", exist_ok=True)
    output_path = os.path.join("joint_csv", f"{last_folder}.csv")

    merged_df.to_csv(output_path, index=False)

    print(f"✅Saved: {output_path}")

# Example usage
if __name__ == "__main__":
    join_all_csvs("/home/cvmsct05/temperatures_max_min/55")
