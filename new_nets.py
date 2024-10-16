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

def eval_net(which, variables, params, save_folder, loss_func, ic):

    if (loss_func == 'task'):
        print("Training wiht task loss")
    elif (loss_func == 'rmse'):
        print("Training with RMSE loss")
    else: 
        print("Error")
        return func

    func = model_classes.ODEFunc()
    #initialize the calib NN, and we set a 
    calib = model_classes.CalibrationNN()
    x = torch.tensor([1.0], device=DEVICE)


    # func.set_beta(0.5)
    y0 = torch.tensor([1938000.0 - ic[0] - ic[1] - 3*ic[2] - 11, ic[0], ic[1], ic[2], ic[2], ic[2], 6.0, 5.0, 0.0, 0.0], dtype=torch.float32)
    # ode_params = list(func.parameters())
    calib_params = list(calib.parameters())
    opt = optim.Adam(calib.parameters(), lr=1e-3)


    for i in range(1000):
        func.train()
        calib.train()
        func.reset_t()
        opt.zero_grad()

        parameters = calib(x)
        print(parameters)
        func.set_params(parameters)

        y0 = torch.tensor([1938000.0 - func.E - func.Ia - func.Ip - func.Im - func.Is - func.Hd - func.Hr, func.E, func.Ia, func.Ip, func.Im, func.Is, func.Hd, func.Hr, 0.0, 0.0], dtype=torch.float32)

        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
        Y_sched_train_total_hospitalizations = (Y_sched_train[:, [6]] + Y_sched_train[:, [7]])
        # print("you are here")
        if (loss_func == 'task'):
            train_loss = task_loss(Y_sched_train_total_hospitalizations.flatten(), torch.cat((variables["Y_train"],variables["Y_test"])), params)
            print(train_loss)
        else: 
            train_loss = rmse_loss(Y_sched_train_total_hospitalizations.flatten(), torch.cat((variables["Y_train"],variables["Y_test"])))
            print(train_loss)


        
        # print(Y_sched_train)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(calib.parameters(), 1e-1)
        opt.step()
        # print('beta:', func.beta)
        # print("you are here")


    func.eval()
    calib.eval()
    func.reset_t()
    parameters = calib(x)
    print(parameters)
    func.set_params(parameters)
    Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
    Y_sched_test_new = Y_sched_test[len(variables['Y_train']):]
    Y_sched_test_total_hospitalizations = Y_sched_test_new[:, 6] + Y_sched_test_new[:, 7]
    # import pdb
    # pdb.set_trace()
    task_test_loss = task_test(Y_sched_test_total_hospitalizations.flatten(), variables['Y_test'], params)
    rmse_test_loss = rmse_test(Y_sched_test_total_hospitalizations.flatten(), variables['Y_test'])

    days = np.arange(1, 8)
    data = np.column_stack((days, task_test_loss.detach(), rmse_test_loss.detach()))

    with open(loss_func + '_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Task Loss', 'RMSE Loss'])
        writer.writerows(data)


    plt.figure()
    plt.plot(np.concatenate((variables["Y_train"],variables["Y_test"])), label = "Data Hospitalizations")
    all_hospitalizations = Y_sched_test[:, 6] + Y_sched_test[: , 7]
    print(all_hospitalizations)
    plt.plot(all_hospitalizations.detach(), label = "Predicted Hospitalizations")
    plt.title("Total Hospitalizations E=" + str(ic[0]) + " Ia=" + str(ic[1]) + " Ip/Im/Is=" + str(ic[2]))
    plt.xlabel("Day")
    plt.ylabel("Hospitalizations")
    plt.legend()
    # plt.savefig("hospitalizations_E=" + str(ic[0]) + "_Ia=" + str(ic[1]) + "_Ip_Im_Is=" + str(ic[2]) + ".png")
    plt.savefig("test.png")



    return func
