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


import model_classes, abm_nets

from constants import *


def main():
    parser = argparse.ArgumentParser(
        description='Run electricity scheduling task net experiments.')
    parser.add_argument('--save', type=str, 
        metavar='save-folder', help='prefix to add to save path')
    parser.add_argument('--nRuns', type=int, default=1,
        metavar='runs', help='number of runs')
    parser.add_argument('--loss', type=str, default="task",
        metavar="loss_function", help="Choose the loss function to train with, use task or rmse")
    args = parser.parse_args()


    base_save = 'results' if args.save is None else '{}-results'.format(args.save)
    for run in range(args.nRuns):
        print (run)

        # save_folder = os.path.join(base_save, str(run))
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)

        params = {"n": 24, "c_ramp": 0.4, "gamma_under": 50, "gamma_over": 0.5, "c_b": 10, "c_h": 1, "q_b": 2, "q_h": 0.5}
            

        initial_conditions = [1000.0, 30000.0, 50.0]
        abm_nets.eval_net(params, args.loss)




if __name__=='__main__':
    main()
