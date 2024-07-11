import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import model_classes
from torchdiffeq import odeint
import math

def task_loss(Y_sched, Y_actual, params):
    # return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
    #         params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
    #         0.5 * (Y_sched - Y_actual)**2).mean()
    under_loss = params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0)
    mse_loss = (Y_sched - Y_actual)**2
    total_loss = (under_loss + over_loss + mse_loss).mean()
    return total_loss
    # return mse_loss.mean()

def rmse_loss(Y_sched, Y_actual):
    # mse_loss = nn.MSELoss()
    # mse = mse_loss(Y_sched, Y_actual)
    # rmse = torch.sqrt(mse)
    diff = Y_sched - Y_actual
    mse = torch.mean(diff ** 2)  # Compute mean squared error
    rmse = torch.sqrt(mse)  # Compute root mean squared error
    return rmse

def negative_log_likelihood(Y_sched, Y_actual, sigma):
    n = len(Y_actual)
    residuals = Y_actual - Y_sched
    nll = 0.5 * n * torch.log(2 * torch.pi * sigma ** 2) + (residuals ** 2 / (2 * sigma ** 2)).sum()
    return nll

#"which" and "save_folder" parameters not used
def eval_net(which, variables, params, save_folder):
    func = model_classes.ODEFunc()

    #Initialize sigma for NLL loss
    sigma = torch.tensor([1.0], requires_grad=True)
    # Lists to store loss values for comparison
    rmse_losses = []
    nll_losses = []
    task_losses = []

    # func.set_beta(0.5)
    y0 = torch.tensor([1938000.0 - 516.0, 0.0, 500.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32)
    ode_params = func.parameters()
    opt = optim.Adam(ode_params, lr=1e-3)
    loss = torch.nn.MSELoss()

    for epoch in range(1000):
        #Where is this coming from?
        func.train()
        func.reset_t()
        opt.zero_grad()
        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_train']), len(variables['Y_train'])), method='rk4')
        Y_sched_train_total_hospitalizations = (Y_sched_train[:, [6]] + Y_sched_train[:, [7]])

        # Y_sched_train_total_hospitalizations = (Y_sched_train[:, 6] + Y_sched_train[:, 7])
        #print("you are here")
        # train_loss = task_loss(Y_sched_train_total_hospitalizations.float(), variables['Y_train'].float(), params)

        Y_sched = Y_sched_train_total_hospitalizations.flatten()
        Y_actual = variables['Y_train']

        train_loss = task_loss(Y_sched, Y_actual, params)
        task_losses.append(train_loss.item())

        # Compute RMSE loss
        rmse = rmse_loss(Y_sched, Y_actual)
        rmse_losses.append(rmse.item())
        # Compute NLL loss
        nll = negative_log_likelihood(Y_sched, Y_actual, sigma)
        nll_losses.append(nll.item())

        # train_loss = loss(Y_sched_train[:, [6]], variables['Y_train'])
        
        # print(Y_sched_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_params, 1e-1)
        opt.step()
        #print('beta:', func.beta)
        #print("you are here")
        
        # with torch.no_grad():
        #     Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
        #     Y_sched_test_new = Y_sched_test[len(variables['Y_train']):]
        #     Y_sched_test_total_hospitalizations = Y_sched_test_new[:, 6] + Y_sched_test_new[:, 7]
        #     # import pdb
        #     # pdb.set_trace()
        #     test_loss = task_loss(Y_sched_test_total_hospitalizations.float(), variables['Y_test'].float(), params)
        
        # print(i, train_loss.item(), test_loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: RMSE = {rmse.item():.4f}, NLL = {nll.item():.4f}, Task Loss = {train_loss.item():.4f}')
        # import pdb
        # pdb.set_trace()

    func.eval()
    

    # Output the final RMSE and NLL values
    print(f'Final RMSE: {rmse.item():.4f}')
    print(f'Final NLL: {nll.item():.4f}')

    # Plot the loss values
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(rmse_losses, label='RMSE Loss')
    plt.plot(nll_losses, label='NLL Loss')
    plt.plot(task_losses, label='Task Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    save_path = '/Users/amums/dl-research/covid/loss_comparison_plot.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    plt.show()
    return func
