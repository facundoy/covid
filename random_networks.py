"""Initializes the agents and the network given files with parameters.
"""
import random
import os
import numpy as np
from collections import Counter
import networkx as nx
from sim_gen_utils import custom_watts_strogatz_graph, get_dir_from_path_list
from itertools import combinations
import pandas as pd
from scipy.stats import nbinom
import yaml
import pdb
import argparse
import random
import numpy as np
import yaml
import os
from tqdm import tqdm


def assign_age_ix_to_agents(age_ix_prob_list, num_agents):
    '''Assigns age index to agents based on the age distribution'''
    res = np.random.choice(len(age_ix_prob_list),
                           p=age_ix_prob_list,
                           size=num_agents)
    return res


def assign_household_ix_to_agents(
    households_sizes_list,
    households_sizes_prob_list,
    num_agents,
    precinct_list,
    precinct_prob_list,
    precinct_votes_prob_list,
):
    '''Assigns household index to agents based on the household size distribution
        Household and precinct should not be independent, 
        so here we also assign a precinct to the household
    '''
    household_id = 0
    total_agents_unassigned = num_agents
    agent_households = []
    household_agents = []
    agent_precincts = []
    agent_votes = []
    last_agent_id = 0
    while total_agents_unassigned > 0:
        household_size = np.random.choice(households_sizes_list,
                                          p=households_sizes_prob_list)
        if household_size > total_agents_unassigned:
            household_size = total_agents_unassigned
        agent_households.extend([household_id] * household_size)
        household_id += 1
        total_agents_unassigned -= household_size
        household_agents.append(
            list(range(last_agent_id, last_agent_id + household_size)))
        last_agent_id += household_size
        # assign a precinct to each household
        precinct = np.random.choice(precinct_list, p=precinct_prob_list)
        agent_precincts.extend([precinct] * household_size)
        # assume all members in household vote for the same candidate
        # assign votes to the household
        agent_votes.extend([
            np.random.choice(['Democrat', 'Republican', 'Others'],
                             p=precinct_votes_prob_list[precinct])
        ] * household_size)

    return agent_households, household_agents, agent_precincts, agent_votes  #precinct_agents


def assign_occupation_ix_to_agents(agents_ages, occupations_sizes_prob_list,
                                   elderly_ix, child_ix, child_upper_ix,
                                   adult_upper_ix):  # from enum
    '''Assigns occupation index to agents based on the age distribution'''
    agents_occupations = []
    for age in agents_ages:
        if age <= child_upper_ix:
            agents_occupations.append(child_ix)
        elif age <= adult_upper_ix:
            agents_occupations.append(
                np.random.choice(len(occupations_sizes_prob_list),
                                 p=occupations_sizes_prob_list))
        else:
            agents_occupations.append(elderly_ix)
    return agents_occupations


def get_num_random_interactions(age, random_network_params_dict,
                                child_upper_ix, adult_upper_ix):
    '''Returns the number of random interactions for an agent based on the age
        Takes the mean and sd from the random_network_params_dict
    '''
    if age <= child_upper_ix:
        mean = random_network_params_dict['CHILD']['mu']
        sd = random_network_params_dict['CHILD']['sigma']
    elif age <= adult_upper_ix:
        mean = random_network_params_dict['ADULT']['mu']
        sd = random_network_params_dict['ADULT']['sigma']
    else:
        mean = random_network_params_dict['ELDERLY']['mu']
        sd = random_network_params_dict['ELDERLY']['sigma']
    
    p = mean / (sd * sd)
    n = mean * mean / (sd * sd - mean)
    num_interactions = nbinom.rvs(n, p)
    return num_interactions


def create_and_write_random_networks(num_agents, agents_ages, num_steps,
                                     random_nw_infile, child_upper_ix,
                                     adult_upper_ix, county, google_param_changes):
    '''Creates and writes the random networks to a file'''
    if not os.path.isfile(random_nw_infile):
        print(
            'The file with random network parameters not found at location {}'.
            format(random_nw_infile))
        raise FileNotFoundError
    
    # Load the network parameters
    random_nw_df = pd.read_csv(random_nw_infile, index_col=0)
    random_network_params_dict = {
        a: {
            'mu': random_nw_df.loc[a, 'mu'],
            'sigma': random_nw_df.loc[a, 'sigma']  # Fixed a typo ('mu' → 'sigma')
        }
        for a in random_nw_df.index.to_list()
    }
    
    # Iterate over timesteps with a progress bar
    for t in range(num_steps):
        # Adjust mu for this timestep
        adjustment_factor = 1.0 + (google_param_changes[t] / 100.0)
        
        for group in random_network_params_dict:
            random_network_params_dict[group]['mu'] *= adjustment_factor

        # Generate random interactions based on updated `mu`
        agents_random_interactions = [
            get_num_random_interactions(age, random_network_params_dict,
                                        child_upper_ix, adult_upper_ix)
            for age in agents_ages
        ]

        interactions_list = []
        for agent_id in range(num_agents):
            interactions_list.extend([agent_id] *
                                     agents_random_interactions[agent_id])
        random.shuffle(interactions_list)
        edges_list = [(interactions_list[i], interactions_list[i + 1])
                      for i in range(len(interactions_list) - 1)]
        G = nx.Graph()
        G.add_edges_from(edges_list)

        outfile_path = f"generated_networks/{county}/specific_networks/randnets"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"random_time_{t}.csv")
        nx.write_edgelist(G, outfile, delimiter=',', data=False)