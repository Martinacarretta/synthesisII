import os
import pandas as pd

def process_csvs(root_path):
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path, dtype=str)  # Read everything as strings
                    changed = False
                    for index, row in df.iterrows():
                        try:
                            max_temp = float(row['Max Temp'])
                            if max_temp > 60:
                                if row['Process'].strip().lower() != 'false':
                                    df.at[index, 'Process'] = 'False'
                                    print(f"Modified: {file_path} | Row: {index}")
                                    changed = True
                        except (ValueError, KeyError):
                            continue
                    if changed:
                        df.to_csv(file_path, index=False)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Example usage
process_csvs("/home/cvmsct05/temperatures_max_min")
