import time

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.logistic_regression.datasets import RegressionDataset
from tensorflow_privacy.privacy.logistic_regression.multinomial_logistic import logistic_dpsgd

from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import compute_epsilon_noise
from scripts.sweep_configuration import sweep_configuration_epsilon_grid


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
    print("Epsilon: " + epsilon)


def main():
    # wandb.login(key=api_key_wandb)

    sweep_id = wandb.sweep(sweep_configuration_epsilon_grid,
                           project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_grid_epsilon, count=10)

    wandb.finish()


if __name__ == '__main__':
    main()
