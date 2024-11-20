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
import random

import warnings
warnings.simplefilter("ignore")
import agent_torch
from agent_torch.models import covid
from agent_torch.models import covid_abm
from agent_torch.populations import astoria
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation


class LearnableParams(nn.Module):
    def __init__(self, num_params, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.fc1 = nn.Linear(1, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, self.num_params).to(self.device)
        self.ReLU = nn.ReLU()
        self.learnable_params = nn.Parameter(torch.rand(num_params, device=self.device))
        self.min_values = torch.tensor([1.5, 1.5, 1.5, 1.5, 0, 0],
                                       device=self.device)
        self.max_values = torch.tensor([6.5, 6.5, 6.5, 6.5, 100, 1],
                                       device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        out = self.fc3(out)
        ''' bound output '''
        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        # out = self.sigmoid(out)
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

    # print("function: ", function)
    # print("index: ", index)
    # print("sub_func: ", sub_func)
    # print("arg_type: ", arg_type)
    # print("var_name: ", var_name)
    
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
    labels = labels.to(DEVICE)
    print(labels)

    reshaped_labels = labels.view(4,7)
    Y_sched = reshaped_labels.sum(dim = 1)
    Y_actual = torch.tensor(Y_actual, dtype=torch.float, device=DEVICE)

    print(Y_sched)


    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared).mean() * len(under_loss)
    return total_loss

def modify_initial_exposed(file_path, proportion):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    header = rows[0]
    data_rows = rows[1:]
    
    flat_data = [0.0 for row in data_rows for item in row]

    total_numbers = len(flat_data)
    num_ones_to_insert = int(total_numbers * proportion)
    
    indices = list(range(total_numbers))
    ones_indices = random.sample(indices, num_ones_to_insert)
    
    for idx in ones_indices:
        flat_data[idx] = 1.0

    reshaped_data = [flat_data[i:i + len(data_rows[0])] for i in range(0, len(flat_data), len(data_rows[0]))]

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(reshaped_data)

def eval_net(params, loss_func):

    if (loss_func == 'task'):
        print("Training wiht task loss")
    elif (loss_func == 'rmse'):
        print("Training with RMSE loss")
    else: 
        print("Error")
        return 
    
    sim = Executor(covid_abm, pop_loader=LoadPopulation(astoria))
    runner = sim.runner
    runner.init()
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]
    
    #Sanity check for cuda
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print("")

    df = pd.read_csv("astoria_data.csv", parse_dates = ["date"])
    case_numbers = df['cases'].values
    case_numbers = torch.tensor(case_numbers, dtype=torch.float, device=DEVICE)

    learn_model = LearnableParams(6, device=DEVICE)
    opt = optim.Adam(learn_model.parameters(), lr=0.01)
    loss_data = []
    x = torch.tensor([1.0], device=DEVICE)
    for i in range(1):
        print("Epoch", i)
        torch.autograd.set_detect_anomaly(True)

        opt.zero_grad()

        runner.reset()
        debug_tensor = learn_model(x)
        print("Debug tensor: ", debug_tensor)
        print("R0: ", debug_tensor[0])
        print("Initial proportion of exposed: ", debug_tensor[5])
        debug_tensor = debug_tensor[:, None]
        
        # set parameters
        # TODO: turn it into a single function
        input_string = learnable_params[0][0]
        print(input_string)
        tensorfunc = map_and_replace_tensor(input_string)
        current_tensor = tensorfunc(runner, debug_tensor[:4], mode_calibrate=True)
        
        #Sanity check for cuda 
        # print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        # print(f"Cached: {torch.cuda.memory_reserved()} bytes")



        input_string = learnable_params[1][0]
        print(input_string)
        tensorfunc = map_and_replace_tensor(input_string)
        current_tensor = tensorfunc(runner, debug_tensor[5], mode_calibrate=True)

        input_string = learnable_params[2][0]
        print(input_string)
        tensorfunc = map_and_replace_tensor(input_string)
        current_tensor = tensorfunc(runner, debug_tensor[4], mode_calibrate=True)
        

        # execute runner
        loss = execute(runner, case_numbers, params)
        print("Loss:", loss)
        loss.backward()
        # compute gradient
        learn_params_grad = [(param, param.grad) for (name, param) in learn_model.named_parameters()]
        opt.step()

        loss_data.append([i, loss.item()])
        print("*********************")

    with open("loss_data.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Loss"])
        writer.writerows(loss_data) 
        # print("Gradients: ", learn_params_grad)
        # print("---"*10)
