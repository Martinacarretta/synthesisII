import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict


# Script used to plot the temperature data of babies. We can plot all variables in one plot or one per variable.

# There are 2 functions:
# 1. process_csv: processes a single CSV file (subfolder), extracting timestamps and plotting all temperature variables.
# 2. plot_per_baby: aggregates data for each baby and plots all variables in one plot and one per variable.

fever = { # Used to create a vertical red dotted line in the plots of "ground truth" diagnosis
    36: 20240924180000, 
    40: 20241025000000, # no posa hora
    44: 20241120000000, # no posa hora
    47: 20241128050000,
    48: 20241126183000
}
baby_data = defaultdict(list) # Global dictionary to store data for each baby

def process_csv(input_csv_path, output_base="/home/cvmsct05/diagnosis/"):
    df = pd.read_csv(input_csv_path)

    parts = input_csv_path.strip("/").split("/")
    baby_id = parts[-3]
    date = parts[-2]

    temp_columns = ['core', 'left_hand', 'right_hand', 'left_foot', 'right_foot']

    # Extract timestamps from image paths
    def extract_timestamp(path):
        try:
            filename = os.path.basename(path)
            ts_str = filename[2:16]  # Skip "HM", grab 14 characters
            return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except:
            return None

    # Preprocess the DataFrame
    df['timestamp'] = df['Image Path'].apply(extract_timestamp)
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by='timestamp')

    # Add to global baby_data for later aggregation
    baby_data[baby_id].append(df)

    # Define relative output directory
    rel_dir = os.path.dirname(input_csv_path).replace("/home/cvmsct05/temperatures_max_min/", "")
    output_dir = os.path.join(output_base, rel_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Plot: All variables
    plt.figure(figsize=(12, 6))
    for col in temp_columns:
        if col in df.columns:
            plt.plot(df['timestamp'], df[col], label=col, marker='o', markersize=2)

    plt.title(f"Temperatures for Baby {baby_id} on {date}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{date}.png")
    plt.savefig(output_path)
    plt.close()


    print(f"✅ Results saved to {output_dir}")


def plot_per_baby(output_base="/home/cvmsct05/diagnosis/"):
    temp_columns = ['core', 'left_hand', 'right_hand', 'left_foot', 'right_foot']

    # Combine data of subfolders for each baby
    for baby_id, df_list in baby_data.items():
        combined_df = pd.concat(df_list)
        combined_df = combined_df.dropna(subset=['timestamp'])
        combined_df = combined_df.sort_values(by='timestamp')

        baby_dir = os.path.join(output_base, baby_id)
        os.makedirs(baby_dir, exist_ok=True)

        # Plot: All variables combined for baby
        plt.figure(figsize=(12, 6))
        for col in temp_columns:
            if col in combined_df.columns:
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, marker='o', markersize=2)

        plt.title(f"Temperatures for Baby {baby_id} (All Dates)")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        
        # Add fever line if exists
        fever_timestamp = fever.get(int(baby_id))
        if fever_timestamp:
            fever_dt = datetime.strptime(str(fever_timestamp), "%Y%m%d%H%M%S")
            plt.axvline(x=fever_dt, color='red', linestyle='--', label='Fever')

        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        all_output_path = os.path.join(baby_dir, f"{baby_id}_all_dates.png")
        plt.savefig(all_output_path)
        plt.close()

        # Plot: One per variable for baby
        for col in temp_columns:
            if col in combined_df.columns:
                
                # Colors
                if col == 'core': 
                    designated_color = 'tab:blue'
                elif col == 'left_hand':
                    designated_color = 'tab:orange'
                elif col == 'right_hand':
                    designated_color = 'tab:green'
                elif col == 'left_foot':
                    designated_color = 'tab:red'
                elif col == 'right_foot':
                    designated_color = 'tab:purple'
                    
                plt.figure(figsize=(12, 6))
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, color=designated_color, marker='o', markersize=2)
                plt.title(f"{col.capitalize()} Temperature for Baby {baby_id} (All Dates)")
                plt.xlabel("Time")
                plt.ylabel("Temperature")
                
                # Add fever line if exists
                fever_timestamp = fever.get(int(baby_id))
                if fever_timestamp:
                    fever_dt = datetime.strptime(str(fever_timestamp), "%Y%m%d%H%M%S")
                    plt.axvline(x=fever_dt, color='red', linestyle='--', label='Fever')

                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                var_output_path = os.path.join(baby_dir, f"{col}.png")
                plt.savefig(var_output_path)
                plt.close()

        print(f"📊 Baby {baby_id} plots saved to {baby_dir}")


def main(input_path): # traverse path and process all csv files
    if input_path.endswith(".csv"):
        process_csv(input_path)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    process_csv(full_path)

    plot_per_baby()
    baby_data.clear()  # 🔁 Reset after each baby's folder is processed
    


if __name__ == "__main__":
    ### ONLY PUT SUBFOLDERS OF BABIES, DO NOT PUT A WHOLE FOLDER OF NEWBORNS
    # main("/home/cvmsct05/diagnosis/32")
    
    for i in range(29, 56):
        folder_path = f"/home/cvmsct05/diagnosis/{i}"
        if os.path.isdir(folder_path):
            main(folder_path)
        else:
            print(f"❌ Skipping missing folder: {folder_path}")

