import pandas as pd
from collections import Counter
import networkx as nx
import numpy as np
import os
from sim_gen_utils import custom_watts_strogatz_graph, normal_watts_strogatz_graph
from isolate_google_mob_data import get_google_mob_data
from tqdm import tqdm


def generate_mobility_networks(state_abbrev, county, output_dir, num_steps):
    # Load individual data
    agent_data_path = f"{state_abbrev}_population_data/{county}_population.csv"
    individuals = pd.read_csv(agent_data_path)

    # Load network parameters
    occ_params_data_path = "network_parameters/occupation_network_parameters.csv"
    occupation_params = pd.read_csv(occ_params_data_path, index_col="Occupation")

    random_params_data_path = "network_parameters/random_network_parameters.csv"
    random_params = pd.read_csv(random_params_data_path, index_col="Age")

    # Map occupation names to indices
    occupation_names = occupation_params.index.to_list()
    occupation_to_ix = {name: idx for idx, name in enumerate(occupation_names)}

    # Filter individuals with valid occupations
    individuals['OccupationID'] = individuals['Occupations'].map(occupation_to_ix)
    individuals_with_occupations = individuals.dropna(subset=['OccupationID'])

    # Group individuals by occupation
    occupation_groups = {
        occ: individuals_with_occupations[individuals_with_occupations['OccupationID'] == occ]['ID'].tolist()
        for occ in occupation_to_ix.values()
    }

    # Count population per occupation
    occupation_population = Counter(individuals_with_occupations['OccupationID'])

    # Group individuals by household
    households = individuals.groupby('Household ID').groups

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("Generating Occupation Network:")
    
    output_occ_dir = f"{output_dir}/occnets"
    os.makedirs(output_occ_dir, exist_ok=True)

    output_school_dir = f"{output_dir}/schoolnets"
    os.makedirs(output_school_dir, exist_ok=True)
    
    output_random_dir = f"{output_dir}/randomnets"
    os.makedirs(output_random_dir, exist_ok=True)

    #Use google's mobility data to adjust mobility parameters (occupation and household):
    google_occdata_2020, google_housedata_2020, google_retaildata_2020, google_grocerydata_2020, google_parksdata_2020, google_transitdata_2020,  google_occdata_2021, google_housedata_2021, google_retaildata_2021, google_grocerydata_2021, google_parksdata_2021, google_transitdata_2021 = get_google_mob_data()
    
    #Occupation data google data setup:
    occ_google_perc_changes_2020 = google_occdata_2020[county]
    occ_google_perc_changes_2021 = google_occdata_2021[county]
    assert len(occ_google_perc_changes_2020) == len(occ_google_perc_changes_2021)
    # Convert lists to NumPy arrays and compute element-wise average
    occ_arr1 = np.array(occ_google_perc_changes_2020)
    occ_arr2 = np.array(occ_google_perc_changes_2021)
    occ_averages = (occ_arr1 + occ_arr2) / 2.0
    # Convert to Python list
    occ_averages_list = occ_averages.tolist()
    # print(occ_averages_list)

    retail_google_perc_changes_2020 = google_retaildata_2020[county]
    retail_google_perc_changes_2021 = google_retaildata_2021[county]
    assert len(retail_google_perc_changes_2020) == len(retail_google_perc_changes_2021)
    retail_arr1 = np.array(retail_google_perc_changes_2020)
    retail_arr2 = np.array(retail_google_perc_changes_2021)
    retail_averages = (retail_arr1 + retail_arr2) / 2.0
    retail_averages_list = retail_averages.tolist()

    grocery_google_perc_changes_2020 = google_grocerydata_2020[county]
    grocery_google_perc_changes_2021 = google_grocerydata_2021[county]
    assert len(grocery_google_perc_changes_2020) == len(grocery_google_perc_changes_2021)
    grocery_arr1 = np.array(grocery_google_perc_changes_2020)
    grocery_arr2 = np.array(grocery_google_perc_changes_2021)
    grocery_averages = (grocery_arr1 + grocery_arr2) / 2.0
    grocery_averages_list = grocery_averages.tolist()

    parks_google_perc_changes_2020 = google_parksdata_2020[county]
    parks_google_perc_changes_2021 = google_parksdata_2021[county]
    assert len(parks_google_perc_changes_2020) == len(parks_google_perc_changes_2021)
    parks_arr1 = np.array(parks_google_perc_changes_2020)
    parks_arr2 = np.array(parks_google_perc_changes_2021)
    parks_averages = (parks_arr1 + parks_arr2) / 2.0
    parks_averages_list = parks_averages.tolist()

    transit_google_perc_changes_2020 = google_transitdata_2020[county]
    transit_google_perc_changes_2021 = google_transitdata_2021[county]
    assert len(transit_google_perc_changes_2020) == len(transit_google_perc_changes_2021)
    transit_arr1 = np.array(transit_google_perc_changes_2020)
    transit_arr2 = np.array(transit_google_perc_changes_2021)
    transit_averages = (transit_arr1 + transit_arr2) / 2.0
    transit_averages_list = transit_averages.tolist()

    house_google_perc_changes_2020 = google_housedata_2020[county]
    house_google_perc_changes_2021 = google_housedata_2021[county]
    assert len(house_google_perc_changes_2020) == len(house_google_perc_changes_2021)
    house_arr1 = np.array(house_google_perc_changes_2020)
    house_arr2 = np.array(house_google_perc_changes_2021)
    house_averages = (house_arr1 + house_arr2) / 2.0
    house_avg = house_averages.mean()
    house_average = float(house_avg)

    # Outer loop for time steps with tqdm
    for t in tqdm(range(num_steps), desc="Time Steps Progress"):
        #Occupation data google adjustment (mu adjustment):
        occ_mu_perc_change = occ_averages_list[t]
        occ_decimal = occ_mu_perc_change / 100.0
        occ_factor = 1.0 + occ_decimal

        # print(f"Occ factor: {occ_factor}")
        # quit()

        # Inner loop for occupation groups with tqdm
        for occ, agents in tqdm(occupation_groups.items(), desc=f"Occupation Groups Progress (Step {t})", leave=False):
            n_agents = len(agents)
            if n_agents > 1:  # Avoid empty or trivial networks
                mu = occupation_params.loc[occupation_names[occ], 'mu']
                mu *= occ_factor
                rewire = occupation_params.loc[occupation_names[occ], 'rewire']
                avg_degree = min(int(np.round(mu)), n_agents - 1)  # Ensure avg_degree < n_agents

                # Generate network using custom implementation
                G = custom_watts_strogatz_graph(
                    n=n_agents,                      # Number of nodes
                    k=avg_degree,                    # Average degree
                    p=[rewire, agents]               # Rewiring probability and node names
                )
                
                # Map agent indices back to original IDs
                id_mapping = {i: agents[i] for i in range(n_agents)}
                G = nx.relabel_nodes(G, id_mapping)
                
                # Save network to file
                outfile = os.path.join(output_occ_dir, f"{occupation_names[occ]}_step_{t}.csv")
                nx.write_edgelist(G, outfile, delimiter=",", data=False)
    
    print()
    print("Generating Household Network:")

    n_agents = len(individuals)
    if n_agents > 1:  # Avoid empty or trivial networks
        # Generate network using custom implementation
        G = custom_watts_strogatz_graph(
            n=n_agents,             # Number of nodes
            k=0,                    # Average degree = 0
            p=[0, agents]           # Rewiring probability = 0 and node names
        )

        # Add edges for agents in the same household
        for household_id, members in households.items():
            # Convert dataframe indices to actual agent IDs
            household_members = individuals.loc[members, 'ID'].tolist()
            
            # Add edges between all pairs of household members
            if len(household_members) > 1:  # Skip single-member households
                for i in range(len(household_members)):
                    for j in range(i + 1, len(household_members)):
                        G.add_edge(household_members[i], household_members[j])
        
        # Save network to file
        outfile = os.path.join(output_dir, f"HOUSEHOLD_NETWORK.csv")
        nx.write_edgelist(G, outfile, delimiter=",", data=False)

    print()
    print("Generating School Network:")

    children = individuals[(individuals['Age'] >= 0) & (individuals['Age'] <= 19)]
    agents = children['ID'].to_list()
    mu = random_params.loc['CHILD', 'mu']
    sigma = random_params.loc['CHILD', 'sigma']

    for t in tqdm(range(num_steps), desc="Time Steps Progress"):
        if n_agents > 1:  # Avoid empty or trivial networks
            # Generate network using custom implementation
            G = normal_watts_strogatz_graph(
                n=n_agents,             # Number of nodes
                agents=agents,          # Agent IDs
                mu=mu,                  # Degree average
                sigma=sigma             # Degree standard deviation
            )
            
            # Save network to file
            outfile = os.path.join(output_school_dir, f"School_step_{t}.csv")
            nx.write_edgelist(G, outfile, delimiter=",", data=False)
    
    print()
    print("Generating Random Network:")

    children = individuals[(individuals['Age'] >= 0) & (individuals['Age'] <= 19)]
    child_agents = children['ID'].to_list()
    mu_child = random_params.loc['CHILD', 'mu']
    sigma_child = random_params.loc['CHILD', 'sigma']

    adults = individuals[(individuals['Age'] >= 20) & (individuals['Age'] <= 64)]
    adult_agents = adults['ID'].to_list()
    mu_adult = random_params.loc['ADULT', 'mu']
    sigma_adult = random_params.loc['ADULT', 'sigma']

    elderly = individuals[(individuals['Age'] >= 65)]
    elderly_agents = elderly['ID'].to_list()
    mu_elderly = random_params.loc['ELDERLY', 'mu']
    sigma_elderly = random_params.loc['ELDERLY', 'sigma']
    
    for t in tqdm(range(num_steps), desc="Time Steps Progress"):
        rand_mu_perc_change = ((retail_averages_list[t] +  parks_averages_list[t] + transit_averages_list[t] + grocery_averages_list[t])) / 4.0
        rand_decimal = rand_mu_perc_change / 100
        rand_factor = 1.0 + rand_decimal
        new_mu_child = mu_child * rand_factor
        new_sigma_child = sigma_child * rand_factor
        num_child_agents = len(child_agents)
        if num_child_agents > 1:  
            G_child = normal_watts_strogatz_graph(
                n=num_child_agents,            
                agents=child_agents,         
                mu=new_mu_child,                  
                sigma=new_sigma_child            
            )
            outfile = os.path.join(output_random_dir, f"Random_child_step_{t}.csv")
            nx.write_edgelist(G_child, outfile, delimiter=",", data=False)

        new_mu_adult = mu_adult * rand_factor
        new_sigma_adult = sigma_adult * rand_factor
        num_adult_agents = len(adult_agents)
        if num_adult_agents > 1: 
            G_adult = normal_watts_strogatz_graph(
                n=num_adult_agents,           
                agents=adult_agents,        
                mu=new_mu_adult,                
                sigma=new_sigma_adult         
            )
            outfile = os.path.join(output_random_dir, f"Random_adult_step_{t}.csv")
            nx.write_edgelist(G_adult, outfile, delimiter=",", data=False)

        new_mu_elderly = mu_elderly * rand_factor
        new_sigma_elderly = sigma_elderly * rand_factor
        num_elderly_agents = len(elderly_agents)
        if num_elderly_agents > 1: 
            G_elderly = normal_watts_strogatz_graph(
                n=num_elderly_agents,            
                agents=elderly_agents,          
                mu=new_mu_elderly,                 
                sigma=new_sigma_elderly          
            )
            outfile = os.path.join(output_random_dir, f"Random_elderly_step_{t}.csv")
            nx.write_edgelist(G_elderly, outfile, delimiter=",", data=False)