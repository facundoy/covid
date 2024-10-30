from agent_torch.data.census.census_loader import CensusDataLoader
import torch.nn as nn
import os
import pandas as pd
import json
import torch
import numpy as np
# import pickle
from numpy.random import choice
import count_people as cntppl
import re

# import ray
# ray.init(address='auto')

AGE_GROUP_MAPPING = {
    "adult_list": ["20t29", "30t39", "40t49", "50t59", "60t69", "70t79", "80A"],  # Age ranges for adults
    "children_list": ["0t9", "10t19"],  # Age range for children
}

MOBILITY_MAPPING = json.load(open('mobility_mapping.json'))

def _initialize_infections(num_agents, save_dir=None, initial_infection_ratio=0.04):
    # figure out initial infection ratio    
    correct_num_cases = False
    num_tries = 0

    num_cases = int(initial_infection_ratio*num_agents)

    if save_dir is None:
        save_path = './disease_stages_{}.csv'.format(num_agents)
    else:
        save_path = os.path.join(save_dir, 'disease_stages.csv')
        
    while not correct_num_cases:
        prob_infected = initial_infection_ratio * torch.ones(
            (num_agents, 1)
        )
        p = torch.hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch.log(p + 1e-9)
        agent_stages = nn.functional.gumbel_softmax(
            logits=cat_logits, tau=1, hard=True, dim=1
        )[:, 0]
        tensor_np = agent_stages.numpy().astype(int)
        tensor_np = np.array(tensor_np, dtype=np.uint8)
        np.savetxt("temp_infections", tensor_np, delimiter='\n')

        # check if the generated file has the correct number of cases
        arr = np.loadtxt(f"temp_infections")
        if arr.sum() == num_cases:
            correct_num_cases = True
            # write the correct file with the line that says stages
            with open(
                save_path,
                "w",
            ) as f:
                f.write("stages\n")
                np.savetxt(f, tensor_np)

        num_tries += 1
        if num_tries >= 1000:
            raise Exception("failed to create disease stages file")

def customize(data_dir, results_dir, county, num_agents = None):
    #My implementation

    #------------------------MANAGING DATA PATHS------------------------
    # Check if the directory exists
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"The directory '{results_dir}' does not exist. Please create it first.")
    
    # Check if data_dir exists
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please ensure it exists and is accessible.")
    
    #County data path
    county_data_dir = os.path.join(data_dir, county)
    if not os.path.isdir(county_data_dir):
        raise FileNotFoundError(f"The file '{county_data_dir}' does not exist. Please ensure it is available in '{data_dir}'.")
        
    #Set data directories:
    age_groups_data_path = os.path.join(county_data_dir, 'agents_ages.csv')
    household_sizes_data_path = os.path.join(county_data_dir, 'agents_household_sizes.pkl')
    occupations_data_path = os.path.join(county_data_dir, 'agents_occupations.csv')

    # Check if each file path exists
    if not os.path.isfile(age_groups_data_path):
        raise FileNotFoundError(f"The file '{age_groups_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")

    if not os.path.isfile(household_sizes_data_path):
        raise FileNotFoundError(f"The file '{household_sizes_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")

    if not os.path.isfile(occupations_data_path):
        raise FileNotFoundError(f"The file '{occupations_data_path}' does not exist. Please ensure it is available in '{county_data_dir}'.")


    #Load age group data
    AGE_GROUPS_DATA = pd.read_csv(age_groups_data_path)
    # Load household data
    HOUSEHOLD_DATA = pd.read_csv(household_sizes_data_path)
    #load occupation data
    OCCUPATION_DATA = pd.read_csv(occupations_data_path)

    #Create list of ages, household, and occupation stats
    ages_list = []
    household_list = []
    occupations_list = []

    #Create list of random ages in order of age groups
    for index, row in AGE_GROUPS_DATA.iterrows():
        lower = None
        upper = None
        age_group = row['Age']
        match = re.findall(r'\d+', age_group)
        # Convert extracted strings to integers
        if len(match) == 2:  # Ensure there are two numbers found
            lower, upper = int(match[0]), int(match[1])

        num_age_group = int(row['Number'])
        for i in range(num_age_group):
            random_int = np.random.randint(lower, upper + 1)
            ages_list.append(random_int)

    #Create list of household sizes in order 
    for index, row in HOUSEHOLD_DATA.iterrows():
        houseSize = row['Household Size']
        num_household_size = int(row['Number'])
        for i in range(num_household_size):
            household_list.append(houseSize)
            
    #Create list of occupations in order
    for index, row in OCCUPATION_DATA.iterrows():
        occupation = str(row['Occupation'])
        num_occupation = int(row['Number'])
        for i in range(num_occupation):
            occupations_list.append(occupation)
    
    #Randomly shuffle the lists to randomize population:
    np.random.shuffle(ages_list)
    np.random.shuffle(household_list)
    np.random.shuffle(occupations_list)

    #------------------------EVENING OUT POPULATION SIZES OF DATASETS------------------------
    largest_pop = cntppl.largest_total_population(data_path=data_dir)

    if largest_pop == "Occupation":
        print("Invalid data, occupation data larger than age/household data!")
        exit(1)

    if largest_pop == "Ages":
        diff = len(ages_list) - len(household_list)
        assert diff < len(ages_list)
        ages_list = ages_list[:-diff]

    elif largest_pop == "Household":
        diff = len(household_list) - len(ages_list)
        assert diff < len(household_list)
        household_list = household_list[:-diff]
    
    else:
        assert largest_pop == "Equal"
    
    assert len(ages_list) == len(household_list)
    pop_size = len(ages_list)
    

    #------------------------CREATING FINAL DATAFRAME------------------------
    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=["ID", "Age", "Household Size", "Occupations"])

    #Create population of ages and household sizes
    for i in range(pop_size):
        age = ages_list[i]
        household_size = household_list[i]
        row_data = [
            i + 1,
            age,
            household_size,
            None
        ]

        df.iloc[i] = row_data

    #------------------------RANDOMLY ASSIGNING OCCUPATIONS TO POPULATION------------------------
    # Count people with age between 18 and 65
    eligible_count = df[(df["Age"] >= 18) and (df["Age"] <= 65)].shape[0]

    # Check if there are enough eligible people for the occupations
    if eligible_count < len(occupations_list):
        raise ValueError("Not enough people between ages 18 and 65 to assign all occupations") 
    
    # Assign occupations to eligible people
    eligible_indices = df[(df["Age"] >= 18) & (df["Age"] <= 65)].index

    # Shuffle the eligible indices to ensure random assignment
    np.random.shuffle(eligible_indices)

    # Assign occupations
    for i, occupation in zip(eligible_indices, occupations_list):
        df.loc[i, "Occupations"] = occupation

    # Assign "Unemployed" to remaining eligible people without an occupation
    for i in eligible_indices[len(occupations_list):]:
        df.loc[i, "Occupations"] = "Unemployed"
    

    #Convert dataframe to csv and store in results_dir
    # Define the path for the CSV file
    file_path = os.path.join(results_dir, f"{county}_population.csv")

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)  # Set index=False to avoid saving row indices

    return
    #--------------------------------------------------------------------------------------------------
    population_data_path = os.path.join(sample_dir, 'comb_pop.pkl')
    household_data_path = os.path.join(sample_dir, 'agents_household_sizes.pkl')

    # Load household data
    HOUSEHOLD_DATA = pd.read_pickle(household_data_path)

    # Load population data
    BASE_POPULATION_DATA = pd.read_pickle(population_data_path)

    population_dir = 'savePopData'

    census_data_loader = CensusDataLoader(n_cpu=8, use_parallel=False, population_dir=population_dir)
    print("Will save at: ", census_data_loader.population_dir)

    if num_agents is not None:
        census_data_loader.generate_basepop(
            input_data=BASE_POPULATION_DATA,  # The population data frame
            region=region,  # The target region for generating base population
            save_path=population_dir,
            num_individuals = num_agents # Saves data for first 100 individuals, from the generated population
        )
    else:
        census_data_loader.generate_basepop(
            input_data=BASE_POPULATION_DATA,  # The population data frame
            region=region,  # The target region for generating base population
            save_path=population_dir
        )

    if use_household:
        census_data_loader.generate_household(
            household_data=HOUSEHOLD_DATA,  # The loaded household data
            household_mapping=AGE_GROUP_MAPPING,  # Mapping of age groups for household composition
            region=region  # The target region for generating households
        )

    census_data_loader.generate_mobility_networks(
        num_steps=2, 
        mobility_mapping=MOBILITY_MAPPING, 
        region=region
    )

    census_data_loader.export(region)

    return census_data_loader.population_dir #savePopData


if __name__ == '__main__':
    sample_dir = os.path.join(os.getcwd(), 'census_scripts/data/27003')
    num_agents = 1000
    region = '27003'
    # # area_selector = ['BK0101']
    # print("Customizing population")
    # pop_save_dir = customize(sample_dir, num_agents=num_agents, region=region)

    # initial_infection_ratio = 0.04
    # print("Initializing infections")
    # save_dir = os.path.join(pop_save_dir, region)
    # _initialize_infections(num_agents, save_dir=save_dir, initial_infection_ratio=initial_infection_ratio)