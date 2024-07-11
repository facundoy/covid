import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import model_classes
from torchdiffeq import odeint
import matplotlib.pyplot as plt

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

def mse_loss(Y_sched, Y_actual):
    error = (Y_actual - Y_sched)**2
    return error.mean()

def eval_net(which, variables, params, save_folder):
    func = model_classes.ODEFunc()
    # func.set_beta(0.5)
    y0 = torch.tensor([1938000.0 - 516.0, 0.0, 500.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32)
    ode_params = func.parameters()
    opt = optim.Adam(ode_params, lr=1e-3)
    loss = torch.nn.MSELoss()

    train_results = torch.zeros(1000)
    mse_train_results = torch.zeros(1000)
    test_results = torch.zeros(1000)
    mse_test_results = torch.zeros(1000)

    for i in range(1000):
        func.train()
        func.reset_t()
        opt.zero_grad()
        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_train']), len(variables['Y_train'])), method='rk4')
        Y_sched_train_total_hospitalizations = (Y_sched_train[:, [6]] + Y_sched_train[:, [7]])
        # print("you are here")
        train_loss = task_loss(Y_sched_train_total_hospitalizations.flatten(), variables['Y_train'], params)
        train_mse = mse_loss(Y_sched_train_total_hospitalizations.flatten(), variables['Y_train'])
        
        train_results[i] = train_loss
        mse_train_results[i] = train_mse

        # train_loss = loss(Y_sched_train[:, [6]], variables['Y_train'])
        
        # print(Y_sched_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_params, 1e-1)
        opt.step()
        # print('beta:', func.beta)
        # print("you are here")

        test_loss = 0
        
        func.eval()
        with torch.no_grad():
            func.reset_t()
            Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
            Y_sched_test_new = Y_sched_test[len(variables['Y_train']):]
            Y_sched_test_total_hospitalizations = Y_sched_test_new[:, 6] + Y_sched_test_new[:, 7]
            # import pdb
            # pdb.set_trace()
            test_loss = task_loss(Y_sched_test_total_hospitalizations.flatten(), variables['Y_test'], params)
            test_mse = mse_loss(Y_sched_test_total_hospitalizations.flatten(), variables['Y_test'])

            test_results[i] = test_loss
            mse_test_results[i] = test_mse
        
        # print(i, train_loss.item(), test_loss.item())
        print(i, train_loss.item(), train_mse.item(), test_loss.item(), test_mse.item())
        # import pdb
        # pdb.set_trace()

    plt.figure()
    plt.plot(train_results.detach().numpy(), label="Task Loss")
    plt.plot(mse_train_results.detach().numpy(), label = "MSE Loss")
    plt.title("Training Set Loss Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('training_set_loss.png')


    plt.figure()
    plt.plot(test_results.detach().numpy(), label="Task Loss")
    plt.plot(mse_test_results.detach().numpy(), label = "MSE Loss")
    plt.title("Test Set Loss Comparison")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('test_set_loss.png')
    return func
