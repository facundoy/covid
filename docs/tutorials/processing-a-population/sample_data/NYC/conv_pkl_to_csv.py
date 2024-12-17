import os
import pandas as pd

# Get the current directory path
current_directory = os.getcwd()

# Loop through each file in the current directory
for filename in os.listdir(current_directory):
    # Check if the file has a .pkl extension
    if filename.endswith('.pkl'):
        # Construct the full file path
        pkl_path = os.path.join(current_directory, filename)
        
        # Load the pickle file
        try:
            data = pd.read_pickle(pkl_path)
            
            # Case 1: If data is a DataFrame
            if isinstance(data, pd.DataFrame):
                # Save directly as CSV
                csv_filename = filename.replace('.pkl', '.csv')
                data.to_csv(os.path.join(current_directory, csv_filename), index=False)
                print(f"Converted {filename} to {csv_filename}")
            
            # Case 2: If data is a dictionary of DataFrames
            elif isinstance(data, dict):
                for key, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        # Create a unique CSV filename for each key
                        csv_filename = f"{filename.replace('.pkl', '')}_{key}.csv"
                        df.to_csv(os.path.join(current_directory, csv_filename), index=False)
                        print(f"Converted {filename} key '{key}' to {csv_filename}")
                    else:
                        print(f"Skipping {filename} key '{key}': Not a DataFrame")
            
            # Case 3: Unsupported type
            else:
                print(f"Skipping {filename}: Unsupported data type ({type(data)})")
        
        except Exception as e:
            print(f"Could not convert {filename}: {e}")
