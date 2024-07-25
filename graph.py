#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def main():
    rmse_data = np.genfromtxt('rmse_results.csv', delimiter=',', skip_header=1)
    task_data = np.genfromtxt('task_results.csv', delimiter=',', skip_header=1)
    best_fitting_data = np.genfromtxt('best_fit_results.csv', delimiter=',', skip_header=1)
    # fitting_data_100 = np.genfromtxt('100_fit_results.csv', delimiter=',', skip_header=1)
    # fitting_data_500 = np.genfromtxt('500_fit_results.csv', delimiter=',', skip_header=1)
    # fitting_data_1000 = np.genfromtxt('1000_fit_results.csv', delimiter=',', skip_header=1)
    # fitting_data_5000 = np.genfromtxt('5000_fit_results.csv', delimiter=',', skip_header=1)
    # fitting_data_10000 = np.genfromtxt('10000_fit_results.csv', delimiter=',', skip_header=1)

    rmse_train_task_test = rmse_data[:, 1]
    rmse_train_rmse_test = rmse_data[:, 2]
    task_train_task_test = task_data[:, 1]
    task_train_rmse_test = task_data[:, 2]
    y_sched_best_data = best_fitting_data[:, 1]
    y_actual_best_data = best_fitting_data[:, 2]
    # y_sched_data_100 = fitting_data_100[:, 1]
    # y_actual_data_100 = fitting_data_100[:, 2]
    # y_sched_data_500 = fitting_data_500[:, 1]
    # y_actual_data_500 = fitting_data_500[:, 2]
    # y_sched_data_1000 = fitting_data_1000[:, 1]
    # y_actual_data_1000 = fitting_data_1000[:, 2]
    # y_sched_data_5000 = fitting_data_5000[:, 1]
    # y_actual_data_5000 = fitting_data_5000[:, 2]
    # y_sched_data_10000 = fitting_data_10000[:, 1]
    # y_actual_data_10000 = fitting_data_10000[:, 2]

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

    plt.figure()
    plt.plot(y_sched_best_data, label="Y_Sched")
    plt.plot(y_actual_best_data, label = "Y_Actual")
    plt.title("Total Hospitalizations: E = 2000.0, I_a = 20000.0, I_p/I_m/I_s = 25.0")
    plt.xlabel("Day")
    plt.ylabel("Hospitalizations")
    plt.legend()
    plt.savefig('best_fit_results.png')

   

    # plt.figure()
    # plt.plot(y_sched_data_100, label="Y_Sched")
    # plt.plot(y_actual_data_100, label = "Y_Actual")
    # plt.title("100 Infected Fit: Prediction vs. Actual")
    # plt.xlabel("Day")
    # plt.ylabel("Hospitalizations")
    # plt.legend()
    # plt.savefig('100_fit_results.png')

    # plt.figure()
    # plt.plot(y_sched_data_500, label="Y_Sched")
    # plt.plot(y_actual_data_500, label = "Y_Actual")
    # plt.title("500 Infected Fit: Prediction vs. Actual")
    # plt.xlabel("Day")
    # plt.ylabel("Hospitalizations")
    # plt.legend()
    # plt.savefig('500_fit_results.png')

    # plt.figure()
    # plt.plot(y_sched_data_1000, label="Y_Sched")
    # plt.plot(y_actual_data_1000, label = "Y_Actual")
    # plt.title("1000 Infected Fit: Prediction vs. Actual")
    # plt.xlabel("Day")
    # plt.ylabel("Hospitalizations")
    # plt.legend()
    # plt.savefig('1000_fit_results.png')

    # plt.figure()
    # plt.plot(y_sched_data_5000, label="Y_Sched")
    # plt.plot(y_actual_data_5000, label = "Y_Actual")
    # plt.title("5000 Infected Fit: Prediction vs. Actual")
    # plt.xlabel("Day")
    # plt.ylabel("Hospitalizations")
    # plt.legend()
    # plt.savefig('5000_fit_results.png')

    # plt.figure()
    # plt.plot(y_sched_data_10000, label="Y_Sched")
    # plt.plot(y_actual_data_10000, label = "Y_Actual")
    # plt.title("10000 Infected Fit: Prediction vs. Actual")
    # plt.xlabel("Day")
    # plt.ylabel("Hospitalizations")
    # plt.legend()
    # plt.savefig('10000_fit_results.png')



if __name__=='__main__':
    main()