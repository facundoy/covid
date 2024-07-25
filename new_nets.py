import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import model_classes
from torchdiffeq import odeint
import math
import sys
import csv

class CalibNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=6):
        super(CalibNN, self).__init__()
        self.min_value = torch.tensor([0.0, 0.4, 0.3, 0.1, 1./20, 1.0 - 0.025])
        self.max_value = torch.tensor([1.0, 0.8, 0.5, 0.3, 1./13, 1.0 - 0.075])
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.fc3(x)
        out = self.min_value + (self.max_value-self.min_value)*self.sigmoid(x)
        return out

def task_train_loss(Y_sched, Y_actual, params):
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

def task_test_loss(Y_sched, Y_actual, params):
    under_loss = params["c_b"] * torch.clamp(Y_actual - Y_sched, min=0)
    over_loss = params["c_h"] * torch.clamp(Y_sched - Y_actual, min=0)
    under_loss_squared = params["q_b"] * torch.clamp((Y_actual - Y_sched)**2, min=0)
    over_loss_squared = params["q_h"] * torch.clamp((Y_sched - Y_actual)**2, min=0)
    total_loss = (under_loss + over_loss + over_loss_squared + under_loss_squared)
    return total_loss


def rmse_train_loss(Y_sched, Y_actual):
    # mse_loss = nn.MSELoss()
    # mse = mse_loss(Y_sched, Y_actual)
    # rmse = torch.sqrt(mse)
    diff = Y_sched - Y_actual
    mse = torch.mean(diff ** 2)  # Compute mean squared error
    rmse = torch.sqrt(mse)  # Compute root mean squared error
    return rmse

def rmse_test_loss(Y_sched, Y_actual):
    diff = Y_sched - Y_actual
    se = diff ** 2 # Compute squared error
    rse = torch.sqrt(se)  # Compute root squared error
    return rse

# def negative_log_likelihood(Y_sched, Y_actual, sigma):
#     n = len(Y_actual)
#     residuals = Y_actual - Y_sched
#     nll = 0.5 * n * torch.log(2 * torch.pi * sigma ** 2) + (residuals ** 2 / (2 * sigma ** 2)).sum()
#     return nll



#"which" and "save_folder" parameters not used
def eval_net(loss, variables, params, infected, y0, save_folder):
    func = model_classes.ODEFunc()

    calib_nn = CalibNN()
    x = torch.tensor([1.0])
    # Get the beta value from the CalibNN

    #Initialize sigma for NLL loss
    # sigma = torch.tensor([1.0], requires_grad=True)
    # Lists to store loss values for comparison
    # rmse_train_losses = []
    # nll_losses = []
    # task_train_losses = []

    # func.set_beta(0.5)
    cal_params = calib_nn.parameters()
    opt = optim.Adam(cal_params, lr=1e-3)
    # loss = torch.nn.MSELoss()

    for epoch in range(1000):
        func.train()
        func.reset_t()
        opt.zero_grad()
        parameters = calib_nn(x)
        func.set_params(parameters)
        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_train']) + len(variables['Y_test']), len(variables['Y_train']) + len(variables['Y_test'])), method='rk4')
        Y_sched_train_total_hospitalizations = (Y_sched_train[:, [6]] + Y_sched_train[:, [7]])

        # Y_sched_train_total_hospitalizations = (Y_sched_train[:, 6] + Y_sched_train[:, 7])
        #print("you are here")
        # train_loss = task_loss(Y_sched_train_total_hospitalizations.float(), variables['Y_train'].float(), params)

        Y_sched = Y_sched_train_total_hospitalizations.flatten()
        Y_actual = torch.cat((variables["Y_train"],variables["Y_test"]))

        if(loss == "task"):
            task_loss = task_train_loss(Y_sched, Y_actual, params)
            # task_train_losses.append(task_loss.item())
        elif(loss == "rmse"):
            rmse_loss = rmse_train_loss(Y_sched, Y_actual)
            # rmse_train_losses.append(rmse_loss.item())
        else:
            sys.stderr.write("Error: Invalid loss function - Must be either 'rmse' or 'task'\n")
            sys.exit(1)
        

        # Compute RMSE loss
        
        # Compute NLL loss
        # nll = negative_log_likelihood(Y_sched, Y_actual, sigma)
        # nll_losses.append(nll.item())

        # train_loss = loss(Y_sched_train[:, [6]], variables['Y_train'])
        
        # print(Y_sched_train)
        if(loss == "rmse"):
            rmse_loss.backward()
        elif(loss == "task"):
            task_loss.backward()

        
        torch.nn.utils.clip_grad_norm_(cal_params, 1e-1)
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
        if epoch % 100 == 0 and loss == "task":
            print(f'Epoch {epoch}: Task Loss = {task_loss.item():.4f}')
            print(f'Beta: {parameters[0].item():.4f}, Ca: {parameters[1].item():.4f}, alpha: {parameters[2].item():.4f}, delta: {parameters[3].item():.4f}, rhoD: {parameters[4].item():.4f}, mu: {parameters[5].item():.4f}')
            # print(f'Y_sched: {Y_sched}, Y_actual: {Y_actual}')
        elif epoch % 100 == 0 and loss == "rmse":
            print(f'Epoch {epoch}: RMSE Loss = {rmse_loss.item():.4f}')
            print(f'Beta: {parameters[0].item():.4f}, Ca: {parameters[1].item():.4f}, alpha: {parameters[2].item():.4f}, delta: {parameters[3].item():.4f}, rhoD: {parameters[4].item():.4f}, mu: {parameters[5].item():.4f}')

        # import pdb
        # pdb.set_trace()

        if(epoch == 999):
            days = np.arange(1, len(variables['Y_train']) + len(variables['Y_test']) + 1)
            data = np.column_stack((days, Y_sched.detach(), Y_actual.detach()))
            with open('best_fit_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Day', 'Y_sched', 'Y_actual'])
                writer.writerows(data)




    func.eval()
    func.reset_t()
    # task_test_loss
    # rmse_test_loss

    # with torch.no_grad():
    Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])), method='rk4')
    Y_sched_test_new = Y_sched_test[len(variables['Y_train']):]
    Y_sched_test_total_hospitalizations = Y_sched_test_new[:, 6] + Y_sched_test_new[:, 7]
    # import pdb
    # pdb.set_trace()
    task_testing_loss = task_test_loss(Y_sched_test_total_hospitalizations.float(), variables['Y_test'].float(), params)
    rmse_testing_loss = rmse_test_loss(Y_sched_test_total_hospitalizations.float(), variables['Y_test'].float())


    # Output the final RMSE and NLL values
    # print(f'Final RMSE Train Loss: {rmse.item():.4f}')
    # print(f'Final Task Train Loss: {train_loss.item():.4f}')

    #Output testing data
    print("")
    print("Here's all the testing data:")
    print("Task Test Loss:")
    print(task_testing_loss)
    print("RMSE Test Loss:")
    print(rmse_testing_loss)

    # # Plot the loss values
    # import matplotlib.pyplot as plt

    days = np.arange(1, 8)
    data = np.column_stack((days, task_testing_loss.detach(), rmse_testing_loss.detach()))

    with open(loss + '_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Task Loss', 'RMSE Loss'])
        writer.writerows(data)

    # plt.figure(figsize=(12, 6))
    # plt.plot(task_testing_loss, label='Task Loss')
    # plt.plot(rmse_testing_loss, label='RMSE Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # save_path = '/Users/amums/dl-research/covid/loss_comparison_plot.png'
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path)

    # plt.show()
    return func
