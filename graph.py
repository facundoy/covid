#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def main():
    rmse_data = np.genfromtxt('rmse_results.csv', delimiter=',', skip_header=1)
    task_data = np.genfromtxt('task_results.csv', delimiter=',', skip_header=1)

    rmse_train_task_test = rmse_data[:, 1]
    rmse_train_rmse_test = rmse_data[:, 2]
    task_train_task_test = task_data[:, 1]
    task_train_rmse_test = task_data[:, 2]

    plt.figure()
    plt.plot(task_train_task_test, label = "Task Loss Trained")
    plt.plot(rmse_train_task_test, label="RMSE Loss Trained")
    plt.title("Task Loss Comparison")
    plt.xlabel("Day")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('task_loss.png')


    plt.figure()
    plt.plot(task_train_rmse_test, label="Task Loss Trained")
    plt.plot(rmse_train_rmse_test, label = "MSE Loss Trained")
    plt.title("RMSE Loss Comparison")
    plt.xlabel("Day")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('rmse_loss.png')



if __name__=='__main__':
    main()