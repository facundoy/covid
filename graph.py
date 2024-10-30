#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def main():
    loss_data = np.genfromtxt('training_loss.csv', delimiter=',', skip_header=1)

    iterations = loss_data[:, 0]
    losses = loss_data[:, 1]

    plt.figure()
    plt.plot(iterations, losses, label="Training Loss")
    plt.title("Training Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig('training_loss_plot.png')



if __name__=='__main__':
    main()