import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import model_classes
from torchdiffeq import odeint

def task_loss(Y_sched, Y_actual, params):
    # return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
    #         params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
    #         0.5 * (Y_sched - Y_actual)**2).mean()
    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared).mean()
    return total_loss
    # return mse_loss.mean()

def eval_net(which, variables, params, save_folder):
    func = model_classes.ODEFunc()
    # func.set_beta(0.5)
    y0 = torch.tensor([1938000.0 - 516.0, 0.0, 500.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32)
    ode_params = func.parameters()
    opt = optim.Adam(ode_params, lr=1e-3)
    loss = torch.nn.MSELoss()

    for i in range(1000):
        func.train()
        func.reset_t()
        opt.zero_grad()
        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_train']), len(variables['Y_train'])), method='rk4')
        Y_sched_train_total_hospitalizations = (Y_sched_train[:, [6]] + Y_sched_train[:, [7]])
        print("you are here")
        train_loss = task_loss(Y_sched_train_total_hospitalizations.flatten(), variables['Y_train'], params)

        # train_loss = loss(Y_sched_train[:, [6]], variables['Y_train'])
        
        # print(Y_sched_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_params, 1e-1)
        opt.step()
        print('beta:', func.beta)
        print("you are here")

        test_loss = 0
        
        with torch.no_grad():
            func.reset_t()
            Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
            Y_sched_test_new = Y_sched_test[len(variables['Y_train']):]
            Y_sched_test_total_hospitalizations = Y_sched_test_new[:, 6] + Y_sched_test_new[:, 7]
            # import pdb
            # pdb.set_trace()
            test_loss = task_loss(Y_sched_test_total_hospitalizations.flatten(), variables['Y_test'], params)
        
        # print(i, train_loss.item(), test_loss.item())
        print(i, train_loss.item(), test_loss.item())
        # import pdb
        # pdb.set_trace()

    func.eval()
    return func
