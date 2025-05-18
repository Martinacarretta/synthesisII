import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

"""
def process_csv (input_csv_path, output_base="/home/cvmsct05/plots/"):
    # Read the CSV
    df = pd.read_csv(input_csv_path)
    print(input_csv_path)
    
    parts = input_csv_path.strip("/").split("/")
    baby_id = parts[-3]
    date = parts[-2]
        
    temp_columns = ['core', 'left_hand', 'right_hand', 'left_foot', 'right_foot']

    # Extract timestamps from image paths
    def extract_timestamp(path):
        try:
            filename = os.path.basename(path)
            # Example: HM20240616163723.jpeg
            ts_str = filename[2:16]  # Skip "HM", grab 14 characters
            return datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        except:
            #print("out of extract_timestamp")
            return None

    df['timestamp'] = df['Image Path'].apply(extract_timestamp)
    # print(df)
    df = df.dropna(subset=['timestamp'])

    # Sort by timestamp
    df = df.sort_values(by='timestamp')

    # # Skip if no data to plot
    # if df.empty:
    #     print(f"⚠️ Skipped: No data to plot for {baby_id} on {date}")
    #     return

    # Plot
    plt.figure(figsize=(12, 6))
    for col in temp_columns:
        if col in df.columns:
            plt.plot(df['timestamp'], df[col], label=col, marker='o', markersize=4)

    plt.title(f"Temperatures for Baby {baby_id} on {date}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()    
    
    # Define relative path from known root, excluding the CSV filename
    rel_dir = os.path.dirname(input_csv_path).replace("/home/cvmsct05/temperatures_max_min/", "")
    output_dir = os.path.join(output_base, rel_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save plot in that directory
    output_path = os.path.join(output_dir, f"{date}.png")
    plt.savefig(output_path)

    plt.close()

    print(f"✅ Results saved to {output_path}")

        
def main (input_path):
    if input_path.endswith(".csv"):
        process_csv(input_path)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".csv"):
                    full_path = os.path.join(root, file)
                    process_csv(full_path)
                    
if __name__ == "__main__":
    main("/home/cvmsct05/output")
    
    
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# Global store for combining data by baby
baby_data = defaultdict(list)

def process_csv(input_csv_path, output_base="/home/cvmsct05/output/"):
    # Read the CSV
    df = pd.read_csv(input_csv_path)
    print(input_csv_path)

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
            plt.plot(df['timestamp'], df[col], label=col, marker='o', markersize=4)

    plt.title(f"Temperatures for Baby {baby_id} on {date}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{date}.png")
    plt.savefig(output_path)
    plt.close()

    # Plot: One per variable
    # for col in temp_columns:
    #     if col in df.columns:
    #         if col == 'core': 
    #             designated_color = 'tab:blue'
    #         elif col == 'left_hand':
    #             designated_color = 'tab:orange'
    #         elif col == 'right_hand':
    #             designated_color = 'tab:green'
    #         elif col == 'left_foot':
    #             designated_color = 'tab:red'
    #         elif col == 'right_foot':
    #             designated_color = 'tab:purple'
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(df['timestamp'], df[col], label=col, color=designated_color, marker='o', markersize=4)
    #         plt.title(f"{col.capitalize()} Temperature for Baby {baby_id} on {date}")
    #         plt.xlabel("Time")
    #         plt.ylabel("Temperature")
    #         plt.legend()
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
    #         var_output_path = os.path.join(output_dir, f"{col}.png")
    #         plt.savefig(var_output_path)
    #         plt.close()

    print(f"✅ Results saved to {output_dir}")


def plot_per_baby(output_base="/home/cvmsct05/output/"):
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
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, marker='o', markersize=3)

        plt.title(f"Temperatures for Baby {baby_id} (All Dates)")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
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
                plt.plot(combined_df['timestamp'], combined_df[col], label=col, color=designated_color, marker='o', markersize=4)
                plt.title(f"{col.capitalize()} Temperature for Baby {baby_id} (All Dates)")
                plt.xlabel("Time")
                plt.ylabel("Temperature")
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


if __name__ == "__main__":
    ### ONLY PUT SUBFOLDERS OF BABIES, DO NOT PUT A WHOLE FOLDER OF NEWBORNS
    main("/home/cvmsct05/output/31")
    main("/home/cvmsct05/output/34")
    main("/home/cvmsct05/output/43")
    main("/home/cvmsct05/output/46")

