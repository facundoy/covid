import pandas as pd
from collections import Counter
import networkx as nx
import numpy as np
import os
from sim_gen_utils import custom_watts_strogatz_graph
from tqdm import tqdm


def generate_mobility_networks(state_abbrev, county, output_dir, num_steps):
    # Load individual data
    agent_data_path = f"{state_abbrev}_population_data/{county}_population.csv"
    individuals = pd.read_csv(agent_data_path)

    # Load network parameters
    occ_params_data_path = "network_parameters/occupation_network_parameters.csv"
    occupation_params = pd.read_csv(occ_params_data_path, index_col="Occupation")

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
    
    # Outer loop for time steps with tqdm
    for t in tqdm(range(num_steps), desc="Time Steps Progress"):
        # Inner loop for occupation groups with tqdm
        for occ, agents in tqdm(occupation_groups.items(), desc=f"Occupation Groups Progress (Step {t})", leave=False):
            n_agents = len(agents)
            if n_agents > 1:  # Avoid empty or trivial networks
                mu = occupation_params.loc[occupation_names[occ], 'mu']
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
                outfile = os.path.join(output_dir, f"{occupation_names[occ]}_step_{t}.csv")
                nx.write_edgelist(G, outfile, delimiter=",", data=False)





