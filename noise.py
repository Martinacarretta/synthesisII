import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from collections import defaultdict

# Script used to plot the temperature data of babies with smoothing. We can plot all variables in one plot or one per variable.
# to do smoothing, we use a rolling mean with a window of 45 samples.

# There are 3 functions:
# 1. noise: plots all variables in one plot and saves it.
# 2. noise2: plots one variable in one plot and saves it.
# 3. plot_per_baby: plots all variables for each baby in one plot and one per variable, saving them in the baby's folder.

baby_data = defaultdict(list) # Global dictionary to store data for each baby

fever = { # Used to create a vertical red dotted line in the plots of "ground truth" diagnosis
    36: 20240924180000, 
    40: 20241025000000, # no posa hora
    44: 20241120000000, # no posa hora
    47: 20241128050000,
    48: 20241126183000
}

TEMP_COLUMNS = ["core","left_hand","right_hand","left_foot","right_foot"]
WINDOW = 45

def noise(csv_path):
    df = pd.read_csv(csv_path)
    
    parts = csv_path.strip("/").split("/")
    baby_id = parts[-3]
    date = parts[-2]

    # To save
    csv_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Internal function to extract timestamp from the filename
    # This function assumes the filename format is "HMYYYYMMDDHHMMSS.jpeg"
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

    # Plot
    plt.figure(figsize=(12, 6))
    for col in TEMP_COLUMNS:
        if col not in df.columns:
            print(f"Column missing in CSV: {col}")
            continue

        smoothed  = (df[col].shift(1).rolling(window=WINDOW, center=False, min_periods = 1).mean()).to_numpy(dtype=float) # Rolling mean with a window of 45 samples
        # smoothed = df[col]
        smoothed_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            col: smoothed
        })

        baby_data[baby_id].append(smoothed_df)

        plt.plot(df['timestamp'], smoothed , label=col, marker='o', markersize=2) #plot each temperature variable
        
    plt.title(f"Smoothed Temperatures for Baby {baby_id} on {date}")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)

    plt.legend()
    plt.tight_layout()

    # Save the figure
    output_name = f"{base_name}_SMOOTHED.png"
    output_path = os.path.join(csv_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {output_path}")
    
    
def noise2(csv_path):
    df = pd.read_csv(csv_path)
    
    parts = csv_path.strip("/").split("/")
    baby_id = parts[-3]
    date = parts[-2]

    # To save
    csv_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Internal function to extract timestamp from the filename
    # This function assumes the filename format is "HMYYYYMMDDHHMMSS.jpeg"
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

    # Plot each temperature variable separately
    for col in TEMP_COLUMNS:
        plt.figure(figsize=(12, 6))
        if col not in df.columns:
            print(f"Column missing in CSV: {col}")
            continue

        smoothed  = (df[col].shift(1).rolling(window=WINDOW, center=False, min_periods = 1).mean()).to_numpy(dtype=float) # rolling mean with a window of 45 samples
        # smoothed = df[col]
        smoothed_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            col: smoothed
        })

        baby_data[baby_id].append(smoothed_df)

        plt.plot(df['timestamp'], smoothed , label=col, marker='o', markersize=2)
        
        plt.title(f"Smoothed Temperatures for Baby {baby_id} on {date}")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.xticks(rotation=45)

        plt.legend()
        plt.tight_layout()

        # Save the figure
        output_name = f"{col}_SMOOTHED.png"
        output_path = os.path.join(csv_dir, output_name)
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")
    
def plot_per_baby(output_base="/home/cvmsct05/diagnosis/"):
    temp_columns = ['core', 'left_hand', 'right_hand', 'left_foot', 'right_foot']

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
                plt.plot(combined_df['timestamp'], combined_df[col], marker='o', label=col, markersize=2)

        plt.title(f"Smoothed Temperatures for Baby {baby_id} (All Dates)")
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
        all_output_path = os.path.join(baby_dir, f"{baby_id}_all_dates_SMOOTH.png")
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
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, marker='o', color=designated_color, markersize=2)
                plt.title(f"Smoothed {col.capitalize()} Temperature for Baby {baby_id} (All Dates)")
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
                var_output_path = os.path.join(baby_dir, f"{col}_SMOOTHED.png")
                plt.savefig(var_output_path)
                plt.close()

        print(f"📊 Baby {baby_id} plots saved to {baby_dir}")


def main(input_path): # traverse path and process all csv files
    if input_path.endswith(".csv"):
        noise(input_path)
        noise2(input_path)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    noise(full_path)
                    noise2(full_path)
    plot_per_baby()
    baby_data.clear()  # 🔁 Reset after each baby's folder is processed


if __name__ == "__main__":
    # main("/home/cvmsct05/diagnosis/32")

    for i in range(29, 56): #for loop for each baby folder
        folder_path = f"/home/cvmsct05/diagnosis/{i}"
        if os.path.isdir(folder_path):
            main(folder_path)
        else:
            print(f"❌ Skipping missing folder: {folder_path}")
