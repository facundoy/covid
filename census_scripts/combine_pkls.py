import os
import pickle
import pandas as pd

# Define the path to the "data" folder
data_folder = "data"

# Function to combine agents_ages.pkl and agents_occupations.pkl
def combine_ages_and_occupations(folder_path):
    # Load the agents_ages.pkl and agents_occupations.pkl files
    ages_pkl_path = os.path.join(folder_path, 'agents_ages.pkl')
    occupations_pkl_path = os.path.join(folder_path, 'agents_occupations.pkl')

    # Load pickle files
    with open(ages_pkl_path, 'rb') as f:
        agents_ages = pickle.load(f)

    with open(occupations_pkl_path, 'rb') as f:
        agents_occupations = pickle.load(f)

    # Combine the data in a way that mimics population.pkl (age_gender + ethnicity)
    # In this case, we'll mimic age_gender as agents_ages and ethnicity as agents_occupations
    combined_data = {
        'ages': agents_ages,         # Mimicking age_gender from population.pkl
        'occupations': agents_occupations    # Mimicking ethnicity from population.pkl
    }

    # Save the combined data into a new pickle file called comb_pop.pkl
    comb_pop_pkl_path = os.path.join(folder_path, 'comb_pop.pkl')
    with open(comb_pop_pkl_path, 'wb') as f:
        pickle.dump(combined_data, f)

    print(f"Combined pickle saved to {comb_pop_pkl_path}")

# Iterate over each folder in the "data" directory
for folder_name in os.listdir(data_folder):
    folder_path = os.path.join(data_folder, folder_name)

    # Check if it's a directory (5-digit folder)
    if os.path.isdir(folder_path) and folder_name.isdigit() and len(folder_name) == 5:
        # Combine the agents_ages.pkl and agents_occupations.pkl into comb_pop.pkl
        combine_ages_and_occupations(folder_path)

print("All folders processed and combined pickle files created.")
