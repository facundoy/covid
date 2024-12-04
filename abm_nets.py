import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import model_classes
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import csv
import pandas as pd
import re

import warnings
warnings.simplefilter("ignore")
import agent_torch
from agent_torch.models import covid
from agent_torch.models import covid_abm
from agent_torch.populations import astoria
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import custom_population as custpop

from gen_mob_nw import generate_mobility_networks

def task_loss(Y_sched, Y_actual, params):
    # return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
    #         params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
    #         0.5 * (Y_sched - Y_actual)**2).mean()
    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared).mean() * len(under_loss)
    return total_loss
    # return mse_loss.mean()

def task_test(Y_sched, Y_actual, params):
    # return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
    #         params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
    #         0.5 * (Y_sched - Y_actual)**2).mean()
    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared)
    return total_loss

def rmse_loss(Y_sched, Y_actual):
    error = (Y_actual - Y_sched)**2
    error = torch.mean(error)
    return torch.sqrt(error)

def rmse_test(Y_sched, Y_actual):
    error = (Y_actual - Y_sched)**2
    return torch.sqrt(error.detach())


class LearnableParams(nn.Module):
    def __init__(self, num_params, device='cpu'):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.fc1 = nn.Linear(self.num_params, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.num_params)
        self.ReLU = nn.ReLU()
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor(2.0,
                                       device=self.device)
        self.max_values = torch.tensor(3.5,
                                       device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        out = self.ReLU(self.fc1(out))
        out = self.ReLU(self.fc2(out))
        beta = self.fc3(out)
        ''' bound output '''
        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        return out

def map_and_replace_tensor(input_string):
    # Split the input string into its components
    parts = input_string.split('.')
    
    # Extract the relevant parts
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    arg_type = parts[4]
    var_name = parts[5]
    
    def getter_and_setter(runner, new_value=None, mode_calibrate=True):
        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)

        if mode_calibrate:
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
        else:
            current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)
        
        if new_value is not None:
            assert new_value.requires_grad == current_tensor.requires_grad
            if mode_calibrate:
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, setvar_name)
            else:
                setvar_name = var_name
                subfunc_param = getattr(substep_function, 'learnable_args')
                setattr(subfunc_param, setvar_name, new_value)
                current_tensor = getattr(subfunc_param, setvar_name)

            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def execute(runner, Y_actual, params, n_steps=28):
    runner.step(n_steps)
    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    # print(labels)

    reshaped_labels = labels.view(4,7)
    Y_sched = reshaped_labels.sum(dim = 1)
    Y_actual = torch.tensor(Y_actual, dtype=torch.float, device=DEVICE)

    print(f"Y_sched: {Y_sched}")
    print(f"Y_actual: {Y_actual}")


    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared).mean() * len(under_loss)
    return total_loss

def eval_net(which, variables, params, save_folder, loss_func, ic):

    if (loss_func == 'task'):
        print("Training with task loss")
    elif (loss_func == 'rmse'):
        print("Training with RMSE loss")
    else: 
        print("Error")
        return 
    
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
    results_path_dir = state_abbrev + '_population_data'
    results_dir = os.path.join(os.getcwd(), results_path_dir)
    # Ensure the results_dir directory exists
    os.makedirs(results_dir, exist_ok=True)

    #Randomly Generate Data Directory
    rand_gen_path_dir = state_abbrev + '_rand_gen_stats_dir'
    rand_gen_dir = os.path.join(os.getcwd(), rand_gen_path_dir)
    # Ensure the rand_gen_dir directory exists
    os.makedirs(rand_gen_dir, exist_ok=True)


    num_agents = 1000

    # Regular expression pattern for a 5-digit FIPS code
    fips_pattern = re.compile(r"^\d{5}$")

    # Iterate through all folders in `sample_dir`
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        # Check if it's a directory and the name matches the FIPS pattern
        if os.path.isdir(folder_path) and fips_pattern.match(folder_name):
            county = folder_name
            # Check if the state code matches the first two digits of the FIPS code
            state_code = int(county[:2])  # Extract first two digits and convert to integer
            if state_code == state_dict[state_abbrev]:
                # Customize population for county
                print(f'Customizing population for county {folder_name}')
                custpop.customize(data_dir=data_dir, results_dir=results_dir, rand_gen_dir=rand_gen_dir, county=county)
    
    #TEST MOBILITY NETWORKS
    print()
    print(f"Generating mobility network...")
    generate_mobility_networks(state_abbrev=state_abbrev, county="26007", output_dir="generated_networks", num_steps=10)

    
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

    #Using custom_population to 
    sim = Executor(covid, pop_loader=LoadPopulation(astoria))
    runner = sim.runner
    runner.init()
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]

    df = pd.read_csv("astoria_data.csv", parse_dates = ["date"])
    case_numbers = df['cases'].values
    # training_data = variables['Y_train']

    loss_array = np.array([])

    learn_model = LearnableParams(3)
    opt = optim.Adam(learn_model.parameters(), lr=1e-3)
    epochs = 1

    for epoch in range(epochs):
        print("Epoch", epoch)
        torch.autograd.set_detect_anomaly(True)

        opt.zero_grad()

        runner.reset()
        debug_tensor = learn_model()[:, None]
        
        # set parameters
        # TODO: turn it into a single function
        input_string = learnable_params[0][0]
        tensorfunc = map_and_replace_tensor(input_string)
        current_tensor = tensorfunc(runner, debug_tensor, mode_calibrate=True)
        # execute runner
        loss = execute(runner, case_numbers, params)
        # print("Loss:", loss)
        loss.backward()
        # compute gradient
        learn_params_grad = [(param, param.grad) for (name, param) in learn_model.named_parameters()]
        opt.step()

        print(f"Loss: {loss}, Loss data type: {type(loss)}")
        loss_np = loss.detach().cpu().numpy()  # Convert to NumPy array
        loss_array = np.append(loss_array, loss_np)
        print(f"Loss_array: {loss_array}")
        # print(f"Loss after backward: {loss}")

        # Print gradients
        # for name, param in learn_model.named_parameters():
        #     print(f"Gradient for {name}: {param.grad}")

        print("*********************")
        # print("Gradients: ", learn_params_grad)
        # print("---"*10)
    
    iters = np.arange(1, epochs + 1)
    print(f"Iters.shape: {iters.shape}, Shape of loss array: {loss_array.shape}")
    assert iters.shape == loss_array.shape
    data = np.column_stack((iters, loss_array))
    with open("training_loss.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Task Loss'])
        writer.writerows(data)

