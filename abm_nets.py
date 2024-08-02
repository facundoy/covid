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

import warnings
warnings.simplefilter("ignore")
import agent_torch
from agent_torch.models import covid
from agent_torch.populations import astoria
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation

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

def execute(runner, Y_actual, params, n_steps=5):
    runner.step(n_steps)
    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']
    print(labels)

    reshaped_labels = labels.view(4, 7)
    Y_sched = reshaped_labels.sum(dim = 1)
    Y_actual = torch.tensor(Y_actual, dtype=torch.float, device=DEVICE)

    print(Y_sched)


    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared).mean() * len(under_loss)
    return total_loss

def eval_net(which, variables, params, save_folder, loss_func, ic):

    if (loss_func == 'task'):
        print("Training wiht task loss")
    elif (loss_func == 'rmse'):
        print("Training with RMSE loss")
    else: 
        print("Error")
        return 
    
    sim = Executor(covid, pop_loader=LoadPopulation(astoria))
    runner = sim.runner
    runner.init()
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]

    df = pd.read_csv("astoria_data.csv", parse_dates = ["date"])
    case_numbers = df['cases'].values

    learn_model = LearnableParams(3)
    for i in range(1000):

        debug_tensor = learn_model()[:, None]
        # set parameters
        input_string = learnable_params[0][0]
        tensorfunc = map_and_replace_tensor(input_string)
        current_tensor = tensorfunc(runner, debug_tensor, mode_calibrate=True)
        # execute runner
        loss = execute(runner, case_numbers, params)
        print(loss)
        loss.backward(retain_graph = True)
        # compute gradient
        learn_params_grad = [(param, param.grad) for (name, param) in learn_model.named_parameters()]
        # print("Gradients: ", learn_params_grad)
        # print("---"*10)

