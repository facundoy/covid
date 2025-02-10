'''Import libraries'''
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import pickle

'''Import functions'''
from gen_mob_nw import generate_mobility_networks
from combine_networks import combine_networks
from custom_population import customize



def save_dict_to_file(dictionary, filename="agent_ages_data.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dictionary, f)

def load_dict_from_file(filename="agent_ages_data.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return {}  # Return an empty dictionary if the file doesn't exist

# Load previous dictionary if it exists
county_to_ageslist_dict = load_dict_from_file()

# Ask user if they want to run the loop
user_input = input("Customize population? (yes/no): ").strip().lower()

if user_input == "yes":
    print("-----------------------------------------------------------------")
    print("Customizing population!")
    print("-----------------------------------------------------------------")
    print()

    # Reset ages list dictionary for new population
    county_to_ageslist_dict = {}

    #CHOOSE STATE
    state_abbrev = 'MI'

    #State abbreviation dictionary
    state_dict = {
        'MI': 26,
        'MN': 27
    }

    #Data Directory
    data_path_dir = 'census_scripts/data'
    data_dir = os.path.join(os.getcwd(), data_path_dir)

    #Results Directory
    results_path_dir = state_abbrev + '_population_data'
    results_dir = os.path.join(os.getcwd(), results_path_dir)
    # Ensure the results_dir directory exists
    os.makedirs(results_dir, exist_ok=True)

    #Randomly Generate Data Directory
    rand_gen_path_dir = state_abbrev + '_rand_gen_stats_dir'
    rand_gen_dir = os.path.join(os.getcwd(), rand_gen_path_dir)
    # Ensure the rand_gen_dir directory exists
    os.makedirs(rand_gen_dir, exist_ok=True)

    # num_agents = 1000

    # Regular expression pattern for a 5-digit FIPS code
    fips_pattern = re.compile(r"^\d{5}$")

    # Iterate through all folders in `data_dir` with a progress bar
    for folder_name in tqdm(os.listdir(data_dir), desc="Processing Counties", unit="folder"):
        folder_path = os.path.join(data_dir, folder_name)
        # Check if it's a directory and the name matches the FIPS pattern
        if os.path.isdir(folder_path) and fips_pattern.match(folder_name):
            county = str(folder_name)
            # Check if the state code matches the first two digits of the FIPS code
            state_code = int(county[:2])  # Extract first two digits and convert to integer
            if state_code == state_dict[state_abbrev]:
                # Customize population for county
                print(f'Customizing population for county {folder_name}...')
                ages_list = customize(data_dir=data_dir, results_dir=results_dir, rand_gen_dir=rand_gen_dir, county=county)
                assert county not in county_to_ageslist_dict, f"Error: '{county}' is already a key in the dictionary!"
                county_to_ageslist_dict[county] = ages_list

    save_dict_to_file(county_to_ageslist_dict)  # Save it for future use

elif not county_to_ageslist_dict:
    assert user_input == "no"
    print("Error: No existing data found. Please customize the population at least once.")
    exit(1)

else:
    assert user_input == "no"
    print("Using previously saved population data.")



# List of Michigan county codes
michigan_county_codes = [
    "001", "003", "005", "007", "009", "011", "013", "015", "017", "019",
    "021", "023", "025", "027", "029", "031", "033", "035", "037", "039",
    "041", "043", "045", "047", "049", "051", "053", "055", "057", "059",
    "061", "063", "065", "067", "069", "071", "073", "075", "077", "079",
    "081", "083", "085", "087", "089", "091", "093", "095", "097", "099",
    "101", "103", "105", "107", "109", "111", "113", "115", "117", "119",
    "121", "123", "125", "127", "129", "131", "133", "135", "137", "139",
    "141", "143", "145", "147", "149", "151", "153", "155", "157", "159",
    "161", "163", "165"
]

# Create a list of Michigan FIPS codes
michigan_fips_codes = ["26" + code for code in michigan_county_codes]

base_dir = "generated_networks"

print()
print("-----------------------------------------------------------------")
print("Generating mobility networks!")
print("-----------------------------------------------------------------")
print()

restart = input("Restart mobility networks for all counties? (yes/no): ").strip().lower()
toRestart = restart == "yes"
if toRestart:
    for fips_code in michigan_fips_codes:
        fips_path = os.path.join(base_dir, fips_code)
        os.makedirs(fips_path, exist_ok=True)
        memory_path = os.path.join(fips_path, "memory.txt")
        # Create an empty text file
        open(memory_path, "w").close()
else:
    assert restart == "no"

num_steps = int(input("Enter number of timesteps: "))
# allCounties = input("Run on all counties? (yes/no): ").strip().lower()
# toRunAllCounties = allCounties == "yes"
county = None

# if not toRunAllCounties:
#     assert allCounties == "no"
#     county = str(input("Enter county FIPS code: "))
#     county_path = os.path.join(base_dir, county)
#     os.makedirs(county_path, exist_ok=True)

#     memory_path = os.path.join(county_path, "memory.txt")
#     # Only create the file if it doesn't exist
#     if not os.path.exists(memory_path):
#         open(memory_path, "w").close()

#     generate_mobility_networks(state_abbrev='MI', county=county, output_dir=county_path, num_steps=num_steps, ages_list=county_to_ageslist_dict[county])
#     # combine_networks(state_abbrev='MI', county=county, output_dir=county_path, num_steps=num_steps)

# else:

# Iterate through Michigan FIPS codes with a progress bar
for fips_code in tqdm(michigan_fips_codes, desc="Processing Counties", unit="county"):
    county_path = os.path.join(base_dir, fips_code)
    os.makedirs(county_path, exist_ok=True)

    memory_path = os.path.join(county_path, "memory.txt")
    # Only create the file if it doesn't exist
    if not os.path.exists(memory_path):
        open(memory_path, "w").close()
    print("-----------------------------------------------------------------")
    generate_mobility_networks(state_abbrev='MI', county=fips_code, output_dir=county_path, num_steps=num_steps, ages_list=county_to_ageslist_dict[fips_code])


print()
print("-----------------------------------------------------------------")
print("Combining all networks!")
print("-----------------------------------------------------------------")
print()

combine_networks(base_dir)

print()
print("Initialization of Experiment Finished!")
