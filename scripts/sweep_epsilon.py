import time

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import compute_epsilon_noise


def sweep_grid_epsilon():
    wandb.init(
        # set the wandb project where this run will be logged
        project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.00001,
            "dataset": "heart_2022_no_nans",
            "epochs": 6,
        }
    )
    """ Prepare and split data"""
    x, y = get_preprocessed_data()

    # Take a look at the number of rows
    # print("Data shape: " + str(x.shape))
    # print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    # print("X_train shape: " + str(x_train.shape))
    # print("X_test shape: " + str(x_test.shape))

    total_samples = x_train.shape[0]
    print("Total samples: ", total_samples)

    epsilon = compute_epsilon_noise(wandb.config.epochs * x_train.shape[0] // wandb.config.batch_size,
                                    wandb.config.noise_multiplier,
                                    wandb.config.batch_size, total_samples)
    wandb.log({"epsilon": epsilon})
    print("Epsilon: " + str(epsilon))


def fetch_epsilon_table():
    x, y = get_preprocessed_data()

    # Take a look at the number of rows
    # print("Data shape: " + str(x.shape))
    # print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    # print("X_train shape: " + str(x_train.shape))
    # print("X_test shape: " + str(x_test.shape))

    total_samples = x_train.shape[0]

    noise_multipliers = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 5,
                         10, 100]

    epsilon_df = pd.DataFrame(columns=['Noise Multiplier', 'Batch Size 15', 'Batch Size 30', 'Batch Size 120'])

    for nm in noise_multipliers:
        epsilon_15 = compute_epsilon_noise(5 * x_train.shape[0] // 15,
                                           nm, 15, total_samples)

        epsilon_30 = compute_epsilon_noise(5 * x_train.shape[0] // 30,
                                           nm, 30, total_samples)

        epsilon_120 = compute_epsilon_noise(5 * x_train.shape[0] // 120,
                                            nm, 120, total_samples)

        new_row = {'Noise Multiplier': nm, 'Batch Size 15': epsilon_15, 'Batch Size 30': epsilon_30,
                   'Batch Size 120': epsilon_120}

        # Inserting the new row
        epsilon_df.loc[len(epsilon_df)] = new_row

    print(epsilon_df.to_latex(index=False))


def plot_epsilon_graphs():
    x, y = get_preprocessed_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    total_samples = x_train.shape[0]

    noise_multiplier = 1.0

    batch_sizes = [15, 150, 1500, 6000]
    epochs = [1, 10, 20, 50, 70, 100]

    epsilon_epochs = pd.DataFrame(columns=['Epsilon', 'Epochs', 'Batch Size'])

    for epoch in epochs:
        for batch_size in batch_sizes:
            epsilon = compute_epsilon_noise(epoch * total_samples // batch_size,
                                            noise_multiplier, batch_size, total_samples)

            new_row = {'Epsilon': epsilon, 'Epochs': epoch, 'Batch Size': batch_size}
            epsilon_epochs.loc[len(epsilon_epochs)] = new_row

    grouped = epsilon_epochs.groupby(['Batch Size'])
    df1 = grouped.get_group(15)
    ax = df1.plot(x="Epochs", y="Epsilon", kind="line",
                  grid=True)

    for key, group in grouped:
        if key != 15:
            group.plot(x="Epochs", y="Epsilon", kind="line",
                       grid=True, ax=ax)
    plt.title("Epsilon dependency on number of epochs and batch size (Noise=1.0)")
    plt.legend(["Batch Size=15", "Batch Size=150", "Batch Size=1500", "Batch Size=6000"], loc='upper left')
    plt.show()


def main():
    # wandb.login(key=api_key_wandb)

    # sweep_id = wandb.sweep(sweep_configuration_epsilon_grid,
    #                        project="sweeps-hyperparameter-tuning-in-privacy-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_grid_epsilon, count=10)
    #
    # wandb.finish()

    fetch_epsilon_table()
    # plot_epsilon_graphs()


if __name__ == '__main__':
    main()
