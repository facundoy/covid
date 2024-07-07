#/usr/bin/env python3

import os
import numpy as np
from constants import *
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

import model_classes

from torchdiffeq import odeint




def task_loss(Y_sched, Y_actual, params):
    return (params["gamma_under"] * torch.clamp(Y_actual - Y_sched, min=0) + 
            params["gamma_over"] * torch.clamp(Y_sched - Y_actual, min=0) + 
            0.5 * (Y_sched - Y_actual)**2).mean(0)

def eval_net(which, variables, params, save_folder):
    # def eval_net(which, model, variables, params, save_folder):
    # model.train()
    func = model_classes.ODEFunc()
    func.set_beta(.5)
    y0 = torch.tensor([1938000.0-10,0,5,5,0,0,0,5,6,0])
    opt = optim.Adam(func.parameters(), lr=1e-3)
    for i in range(1000):
        # model.train()
        opt.zero_grad()
        Y_sched_train = odeint(func, y0, torch.linspace(0, len(variables['Y_train']), len(variables['Y_train'])))
        Y_sched_train_total_hospitalizations = torch.add(Y_sched_train[:,6], Y_sched_train[:,7])
        # import pdb 
        # pdb.set_trace()
        train_loss = task_loss(Y_sched_train_total_hospitalizations.clone().float(), variables['Y_train'].clone(), params)
        # torch.autograd.set_detect_anomaly(True)

        train_loss.backward()
        opt.step()
        
        # model.eval()
        with torch.no_grad():
            Y_sched_test = odeint(func, y0, torch.linspace(0, len(variables['Y_test']) + len(variables['Y_train']), len(variables['Y_test']) + len(variables['Y_train'])))
            Y_sched_test_total_hospitalizations = Y_sched_test[:,6].clone() + Y_sched_test[:,7].clone()
            test_loss = task_loss(Y_sched_test_total_hospitalizations.clone().float(), variables['Y_test'].clone(), params)
        print(i, train_loss.item(), test_loss.item())
    func.eval()

    return func

