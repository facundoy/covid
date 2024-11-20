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


def create_and_write_household_network(household_agents, path):
    '''Creates and writes the household network to a file'''
    household_network = nx.Graph()
    # Adding edges for all agents in each household
    for household in household_agents:
        h_edges = list(combinations(household, 2))
        household_network.add_edges_from(h_edges)
    outfile = os.path.join(get_dir_from_path_list(path), 'household.csv')
    nx.write_edgelist(household_network, outfile, delimiter=',', data=False)


def create_and_write_occupation_networks(agents_occupations,
                                         occupations_names_list,
                                         occupations_ix_list, num_steps,
                                         occupation_nw_infile, path):
    '''Creates and writes the occupation networks to a file'''
    if not os.path.isfile(occupation_nw_infile):
        print(
            'The file with random network parameters not found at location {}'.
            format(occupation_nw_infile))
        raise FileNotFoundError
    occupation_nw_df = pd.read_csv(occupation_nw_infile, index_col=0)
    occupation_nw_parameters_dict = {
        a: {
            'mu': occupation_nw_df.loc[a, 'mu'],
            'rewire': occupation_nw_df.loc[a, 'rewire']
        }
        for a in occupation_nw_df.index.to_list()
    }
    occupations_population = Counter(agents_occupations)
    occupations_agents = [[
        a for a in range(len(agents_occupations)) if agents_occupations[a] == o
    ] for o in occupations_ix_list]
    for t in range(num_steps):
        occupation_networks = {}
        for occ in occupations_ix_list:
            n_interactions = occupation_nw_parameters_dict[
                occupations_names_list[occ]]['mu']
            network_rewire = occupation_nw_parameters_dict[
                occupations_names_list[occ]]['rewire']
            occupation_networks[occ] = custom_watts_strogatz_graph(
                occupations_population[occ],
                min(
                    np.round(n_interactions, 0).astype(int),
                    occupations_population[occ] - 1),
                [network_rewire, occupations_agents[occ]])
        for key in occupations_ix_list:
            outfile = os.path.join(
                get_dir_from_path_list(path + [occupations_names_list[key]]),
                '{}.csv'.format(t))
            G = occupation_networks[key]
            nx.write_edgelist(G, outfile, delimiter=',', data=False)


def create_and_write_random_networks(num_agents, agents_ages, num_steps,
                                     random_nw_infile, child_upper_ix,
                                     adult_upper_ix, path):
    '''Creates and writes the random networks to a file'''
    if not os.path.isfile(random_nw_infile):
        print(
            'The file with random network parameters not found at location {}'.
            format(random_nw_infile))
        raise FileNotFoundError
    random_nw_df = pd.read_csv(random_nw_infile, index_col=0)
    random_network_params_dict = {
        a: {
            'mu': random_nw_df.loc[a, 'mu'],
            'sigma': random_nw_df.loc[a, 'mu']
        }
        for a in random_nw_df.index.to_list()
    }
    agents_random_interactions = [
        get_num_random_interactions(age, random_network_params_dict,
                                    child_upper_ix, adult_upper_ix)
        for age in agents_ages
    ]
    for t in range(num_steps):
        interactions_list = []
        for agent_id in range(num_agents):
            interactions_list.extend([agent_id] *
                                     agents_random_interactions[agent_id])
        random.shuffle(interactions_list)
        edges_list = [(interactions_list[i], interactions_list[i + 1])
                      for i in range(len(interactions_list) - 1)]
        G = nx.Graph()
        G.add_edges_from(edges_list)
        outfile = os.path.join(get_dir_from_path_list(path),
                               '{}.csv'.format(t))
        nx.write_edgelist(G, outfile, delimiter=',', data=False)


def initialize_agents_networks(params):
    '''
        Reads initialization_params.yaml file and initializes agents and networks
        based on the files specified in section initialization_input_files 
        
        This contains agent information such as age, occupation,
            household size distribution, precinct, and voting.
        
        There are 2 more input files for occupation and random network 
            construction parameters.
        
        In addition to creating files in generated_agents_data and generated_networks
        this function also creates a generated_params.yaml file 
        which will be used by main and model
    '''

    # age distribution data for county
    infile = os.path.join(
        params['initialization_input_files']['parent_dir'], params['county'],
        params['initialization_input_files']['agents_ages_filename'])
    if not os.path.isfile(infile):
        print(infile)
        print(
            'The distribution of ages across population file not found at location {}'
            .format(infile))
        raise FileNotFoundError
    ages_df = pd.read_csv(infile, index_col=0)
    params['age_groups'] = ages_df.index.to_list()
    params['age_groups_to_ix_dict'] = {
        params['age_groups'][i]: i
        for i in range(len(params['age_groups']))
    }
    params['age_ix_to_group_dict'] = {
        params['age_groups_to_ix_dict'][k]: k
        for k in params['age_groups_to_ix_dict'].keys()
    }
    total_population_in_age_dist_params = ages_df['Number'].sum()
    params['age_ix_pop_dict'] = {
        k: int(ages_df.loc[params['age_ix_to_group_dict'][k]].values[0])
        for k in range(len(params['age_groups']))
    }
    params['age_ix_prob_list'] = [
        float(params['age_ix_pop_dict'][k] /
              total_population_in_age_dist_params)
        for k in range(len(params['age_groups']))
    ]
    params['num_agents'] = int(ages_df['Number'].sum())

    # household size distribution data for county
    infile = os.path.join(
        params['initialization_input_files']['parent_dir'], params['county'],
        params['initialization_input_files']
        ['agents_household_sizes_filename'])
    if not os.path.isfile(infile):
        print(
            'The file with household sizes distribution not found at location {}'
            .format(infile))
        raise FileNotFoundError
    households_df = pd.read_csv(infile, index_col=0)
    total_population_in_household_dist_params = households_df['Number'].sum()
    params['households_sizes_list'] = households_df.index.to_list()
    params['households_sizes_prob_list'] = [
        float(households_df.loc[k].values[0] /
              total_population_in_household_dist_params)
        for k in params['households_sizes_list']
    ]

    # occupation distribution for county
    infile = os.path.join(
        params['initialization_input_files']['parent_dir'], params['county'],
        params['initialization_input_files']['agents_occupations_filename'])
    if not os.path.isfile(infile):
        print(
            'The file with occupations distribution not found at location {}'.
            format(infile))
        raise FileNotFoundError
    occupations_df = pd.read_csv(infile, index_col=0)
    total_population_in_occupation_dist_params = occupations_df['Number'].sum()
    params['occupations_names_list'] = occupations_df.index.to_list()
    params['occupations_sizes_prob_list'] = [
        float(occupations_df.loc[k].values[0] /
              total_population_in_occupation_dist_params)
        for k in params['occupations_names_list']
    ]
    params['occupations_to_ix_dict'] = {
        i: params['occupations_names_list'][i]
        for i in range(len(params['occupations_names_list']))
    }
    elderly_ix = len(params['occupations_names_list'])
    child_ix = len(params['occupations_names_list']) + 1
    params['occupations_names_list'].extend(['ELDERLY', 'CHILD'])
    params['occupations_to_ix_dict'][elderly_ix] = 'ELDERLY'
    params['occupations_to_ix_dict'][child_ix] = 'CHILD'
    params['occupation_ix_to_occupations_dict'] = {
        params['occupations_to_ix_dict'][k]: k
        for k in params['occupations_to_ix_dict'].keys()
    }
    params['occupations_ix_list'] = list(
        range(len(params['occupations_names_list'])))

    # precinct & voting data for county
    # assumes the file contains percentage instead of total population
    infile = os.path.join(
        params['initialization_input_files']['parent_dir'], params['county'],
        params['initialization_input_files']['agents_precinct_filename'])
    if not os.path.isfile(infile):
        print('The file with precinct distribution not found at location {}'.
              format(infile))
        raise FileNotFoundError
    precinct_df = pd.read_csv(infile, index_col=0)
    params['precinct_list'] = precinct_df.index.to_list()
    params['precinct_prob_list'] = precinct_df['Population'].values.tolist()
    # precinct_votes_prob_list is a dictionary of precinct to list of probabilities
    params['precinct_votes_prob_list'] = {
        p: [
            precinct_df.loc[p,
                            'Democrat'].item(),  # item() to avoid numpy.int64
            precinct_df.loc[p, 'Republican'].item(),
            precinct_df.loc[p, 'Others'].item()
        ]
        for p in params['precinct_list']
    }

    # Creating agent dataframe
    agents_households, household_agents, agent_precincts, agent_votes = assign_household_ix_to_agents(
        params['households_sizes_list'], params['households_sizes_prob_list'],
        params['num_agents'], params['precinct_list'],
        params['precinct_prob_list'], params['precinct_votes_prob_list'])
    agent_df = pd.DataFrame()
    agent_df['age_group'] = assign_age_ix_to_agents(params['age_ix_prob_list'],
                                                    params['num_agents'])
    agent_df['household'] = agents_households
    agent_df['occupation_network'] = assign_occupation_ix_to_agents(
        agent_df['age_group'].values.tolist(),
        params['occupations_sizes_prob_list'], elderly_ix, child_ix,
        params['CHILD_Upper_Index'], params['ADULT_Upper_Index'])
    agent_df['precinct'] = agent_precincts
    agent_df['vote'] = agent_votes

    outfile = os.path.join(
        get_dir_from_path_list([
            params['output_location']['parent_dir'], params['county'],
            params['output_location']['agents_dir']
        ]), params['output_location']['agents_outfile'])
    agent_df.to_csv(outfile)

    # Creating household dataframe
    household_df = pd.DataFrame()
    household_df['Members'] = household_agents
    outfile = os.path.join(
        get_dir_from_path_list([
            params['output_location']['parent_dir'], params['county'],
            params['output_location']['agents_dir']
        ]), params['output_location']['households_outfile'])
    household_df.to_csv(outfile)

    # Constructing networks (assuming agents are ids. Note that ids range from
    # 0 to num_agents-1)
    create_and_write_household_network(household_agents, [
        params['output_location']['parent_dir'], params['county'],
        params['output_location']['networks_dir'],
        params['output_location']['household_networks_dir']
    ])
    create_and_write_occupation_networks(
        agent_df['occupation_network'].values.tolist(),
        params['occupations_names_list'], params['occupations_ix_list'],
        params['num_steps'], params['initialization_input_files']
        ['occupation_nw_parameters_filename'], [
            params['output_location']['parent_dir'], params['county'],
            params['output_location']['networks_dir'],
            params['output_location']['occupation_networks_dir']
        ])
    create_and_write_random_networks(
        params['num_agents'], agent_df['age_group'].values.tolist(),
        params['num_steps'],
        params['initialization_input_files']['random_nw_parameters_filename'],
        params['CHILD_Upper_Index'], params['ADULT_Upper_Index'], [
            params['output_location']['parent_dir'], params['county'],
            params['output_location']['networks_dir'],
            params['output_location']['random_networks_dir']
        ])

    # remove num_steps because it limits the simulator forward pass
    params.pop('num_steps', None)

    # Write updated params file
    params['type'] = 'generated'
    outfile = os.path.join(
        get_dir_from_path_list([params['output_location']['parent_dir']]),
        params['county'], params['genrerated_params_file_name'])
    with open(outfile, 'w') as stream:
        try:
            yaml.dump(params, stream)
        except yaml.YAMLError as exc:
            print(exc)


# Parsing command line arguments
parser = argparse.ArgumentParser(description='Initialize agents and networks')
parser.add_argument('-p',
                    '--params',
                    help='Name of the yaml file with the parameters.')
parser.add_argument('-c', '--county', help='FIPS code for county.')
args = parser.parse_args()

#Reading params
with open(args.params, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print('Error in reading parameters file')
        print(exc)

params['county'] = args.county

# Setting seed
print('Seed used for python random and numpy is {}'.format(params['seed']))
random.seed(params['seed'])
np.random.seed(params['seed'])
pd.np.random.seed(params['seed'])

# Initializing agents and building networks
if params['type'] == "initialization":
    print("Calling initialization..")

    initialize_agents_networks(params)
    # initialize_agents_networks(params)

    print("Initialization done.. ", " exiting")
    exit()
