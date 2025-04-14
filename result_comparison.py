#### THIS IS JUST TO COMPARE THAT THE CODE I MODIFIED FROM CV LOADING TO PIL LOADING WORKED AND BOTH HAD THE SAME RESULTS

import pandas as pd

def compare_csv_files_pandas(file1_path, file2_path):
    """
    Compare two CSV files using pandas.
    
    Args:
        file1_path (str): Path to the first CSV file
        file2_path (str): Path to the second CSV file
        
    Returns:
        bool: True if files are identical, False otherwise
    """
    try:
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        return df1.equals(df2)
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Example usage
if compare_csv_files_pandas('result1.csv', 'result2.csv'):
    print("The CSV files are identical.")
else:
    print("The CSV files are NOT identical.")