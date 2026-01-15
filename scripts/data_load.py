# scripts/01_data_loading.py

import pandas as pd

def load_steel_data(file_path):
    """
    Load steel production dataset
    Parameters:
        file_path (str): Path to the CSV file
    Returns:
        pandas.DataFrame: Loaded data table
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {file_path}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None

if __name__ == "__main__":
    pass