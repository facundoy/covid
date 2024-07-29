#/usr/bin/env python3

import argparse



import os
import pandas as pd
import numpy as np

from datetime import datetime as dt
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar

try: import setGPU
except ImportError: pass

import torch
import torch
import torch.nn as nn
import torch.optim as optim


import model_classes, new_nets

from constants import *



# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode='Verbose',
#      color_scheme='Linux', call_pdb=1)

def main():
    parser = argparse.ArgumentParser(
        description='Run covid bed allocation experiment.')
    parser.add_argument('--save', type=str, 
        metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=1,
        metavar='runs', help='number of runs')
    parser.add_argument('--loss', type=str, default="task")
    args = parser.parse_args()

    # Train, test split.

    df = pd.read_csv("data.csv", parse_dates = ["date"])
    date_list = df['date'].values
    hospitalization_number = df['hosp'].values

    splitting_data_point = int(len(hospitalization_number) * .8)
    training_data = hospitalization_number[:splitting_data_point]
    testing_data = hospitalization_number[splitting_data_point:]
    Y_train = torch.tensor(training_data, dtype=torch.float, device=DEVICE)
    Y_test = torch.tensor(testing_data, dtype=torch.float, device=DEVICE)
    
    variables = {'Y_train': Y_train, 'Y_test': Y_test}

    base_save = 'results' if args.save is None else '{}-results'.format(args.save)

    # print(args.nRuns)
    loss = args.loss

    for run in range(args.nRuns):
        # print (run)

        save_folder = os.path.join(base_save, str(run))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Generation scheduling problem params.
        params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5, "c_b": 10, "c_h": 1, "q_b": 2, "q_h": 0.5}

        # Run and eval rmse-minimizing net
        
        # if USE_GPU:
        #     model_rmse = model_rmse.cuda()

        # y0_dict = {
        #     100: torch.tensor([1938000.0 - 116.0, 0.0, 100.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32),
        #     500: torch.tensor([1938000.0 - 516.0, 0.0, 500.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32),
        #     1000: torch.tensor([1938000.0 - 1016.0, 0.0, 1000.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32),
        #     5000: torch.tensor([1938000.0 - 5016.0, 0.0, 5000.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32),
        #     10000: torch.tensor([1938000.0 - 10016.0, 0.0, 10000.0, 5.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0], dtype=torch.float32)
        # }
        # for key in sorted(y0_dict.keys()):
        #     new_nets.eval_net(loss, variables, params, key, y0_dict[key], save_folder)
        
        new_nets.eval_net(loss, variables, params, save_folder)





if __name__=='__main__':
    main()
