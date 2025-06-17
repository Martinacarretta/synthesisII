import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Global store for combining data by baby
fever = {
    36: 20240924180000, 
    40: 20241025000000, # no posa hora
    44: 20241120000000, # no posa hora
    47: 20241128050000,
    48: 20241126183000
}
baby_data = defaultdict(list)

def process_csv(input_csv_path, output_base="/home/cvmsct05/VALIDATION/plots_validation/"):
    df = pd.read_csv(input_csv_path)

    # Extract baby_id from folder name like "some_47"
    parts = input_csv_path.strip("/").split("/")
    baby_id = parts[-2].split("_")[-1]  # 'some_47' ? '47'

    # Extract date from filename like '01.12.24(1)_VIS_with_temps.csv'
    filename = os.path.basename(input_csv_path)
    raw_date = filename.split("_")[0]  # '01.12.24(1)'
    raw_date_clean = raw_date.split("(")[0]  # '01.12.24'
    date = raw_date_clean.replace(".", "-")  # Convert to '01-12-24'

    temp_columns = ['torso_temp', 'left_hand_temp', 'right_hand_temp', 'left_foot_temp', 'right_foot_temp']


    def extract_timestamp(path):
        try:
            filename = os.path.basename(path)
            ts_str = filename[2:16]  # Skip "HM", grab 14 characters
            return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except:
            return None

    df['timestamp'] = df['Image Path'].apply(extract_timestamp)
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by='timestamp')

    baby_data[baby_id].append(df)

    # Output path
    output_dir = os.path.join(output_base, f"some_{baby_id}")
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"Results saved to {output_dir}")


def plot_per_baby(output_base="/home/cvmsct05/VALIDATION/plots_validation/"):
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
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, marker='o', markersize=2)

        plt.title(f"Temperatures for Baby {baby_id} (All Dates)")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
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


def main(input_path):
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
    # Process all CSVs directly in this folder and its subfolders
    main("/home/cvmsct05/VALIDATION/some_47")


