import os
import pandas as pd

# Define the path to the "data" folder
data_folder = "data"

# Function to convert a CSV file to a pickle file
def convert_csv_to_pickle(csv_path, pkl_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)
    
    # Save the DataFrame as a pickle file
    df.to_pickle(pkl_path)

# Iterate over each folder in the "data" directory
for folder_name in os.listdir(data_folder):
    # Construct the full folder path
    folder_path = os.path.join(data_folder, folder_name)
    
    # Check if it's a directory (a 5-digit folder in this case)
    if os.path.isdir(folder_path) and folder_name.isdigit() and len(folder_name) == 5:
        # File paths for the 3 CSV files in each 5-digit folder
        agents_ages_csv = os.path.join(folder_path, "agents_ages.csv")
        agents_household_sizes_csv = os.path.join(folder_path, "agents_household_sizes.csv")
        agents_occupations_csv = os.path.join(folder_path, "agents_occupations.csv")

        # Corresponding file paths for the pickle files
        agents_ages_pkl = os.path.join(folder_path, "agents_ages.pkl")
        agents_household_sizes_pkl = os.path.join(folder_path, "agents_household_sizes.pkl")
        agents_occupations_pkl = os.path.join(folder_path, "agents_occupations.pkl")

        # Convert CSV files to pickle files
        convert_csv_to_pickle(agents_ages_csv, agents_ages_pkl)
        convert_csv_to_pickle(agents_household_sizes_csv, agents_household_sizes_pkl)
        convert_csv_to_pickle(agents_occupations_csv, agents_occupations_pkl)

        print(f"Converted files in folder {folder_name} to pickle format.")

print("Conversion complete.")
