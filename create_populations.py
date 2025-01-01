import os
import numpy
import re
import custom_population as custpop
from gen_mob_nw import generate_mobility_networks


def create_populations():
    print()
    print("--------CUSTOM POPULATION TESTING--------")

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
    results_path_dir = 'data/' + state_abbrev + '_population_data'
    results_dir = os.path.join(os.getcwd(), results_path_dir)
    # Ensure the results_dir directory exists
    os.makedirs(results_dir, exist_ok=True)

    #Randomly Generate Data Directory
    rand_gen_path_dir = 'data/' + state_abbrev + '_rand_gen_stats_dir'
    rand_gen_dir = os.path.join(os.getcwd(), rand_gen_path_dir)
    # Ensure the rand_gen_dir directory exists
    os.makedirs(rand_gen_dir, exist_ok=True)


    num_agents = 1000

    # Regular expression pattern for a 5-digit FIPS code
    fips_pattern = re.compile(r"^\d{5}$")

    # Iterate through all folders in `sample_dir`
    # for folder_name in os.listdir(data_dir):
    #     folder_path = os.path.join(data_dir, folder_name)
    #     # Check if it's a directory and the name matches the FIPS pattern
    #     if os.path.isdir(folder_path) and fips_pattern.match(folder_name):
    #         county = folder_name
    #         # Check if the state code matches the first two digits of the FIPS code
    #         state_code = int(county[:2])  # Extract first two digits and convert to integer
    #         if state_code == state_dict[state_abbrev]:
    #             # Customize population for county
    #             print(f'Customizing population for county {folder_name}')
    #             custpop.customize(data_dir=data_dir, results_dir=results_dir, rand_gen_dir=rand_gen_dir, county=county)
    
    #TEST MOBILITY NETWORKS
    print()
    print(f"Generating mobility networks for all counties...")

    for folder_name in os.listdir(data_dir):
        county = folder_name
        print(f"Generating mobility network for county {county}...")
        generate_mobility_networks(state_abbrev=state_abbrev, county=county, output_dir="generated_networks", num_steps=10)

    
    quit() #Temporary quit for customize testing

    initial_infection_ratio = 0.04
    print("Initializing infections")
    save_dir = os.path.join(pop_save_dir, region)
    custpop._initialize_infections(num_agents, save_dir=save_dir, initial_infection_ratio=initial_infection_ratio)

    quit()

    # print()
    # print("--------FOLKTABLES TESTING PART--------")
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # acs_data = data_source.get_data(states=["AL"], download=True)
    # features, label, group = ACSEmployment.df_to_numpy(acs_data)
    



    # # Set up the data source for 2020 data with a 1-year horizon for Michigan
    # data_source = ACSDataSource(survey_year='2020', horizon='1-Year', survey='person')

    # # Pull the data for Michigan (MI)
    # acs_data = data_source.get_data(states=["MI"], download=True)

    # # Extract features, labels, and group for the ACSEmployment task
    # features, label, group = ACSEmployment.df_to_numpy(acs_data)

    # # You can now use the features and labels for your machine learning tasks
    # print(features[:5])  # Print first 5 feature rows
    # print(label[:5])     # Print first 5 labels


    # print(f"Features Type: {features.shape}")
    # print()
    # quit()



create_populations()