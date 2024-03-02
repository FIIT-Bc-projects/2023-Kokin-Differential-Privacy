import time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_privacy import logistic_dpsgd
from tensorflow_privacy.privacy.keras_models import dp_keras_model
from tensorflow_privacy.privacy.logistic_regression.datasets import RegressionDataset

from scripts.api_key_wandb import api_key_wandb
from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import get_layers_Binary_Classification, compute_epsilon_noise
from scripts.sweep_configuration import sweep_configuration_grid_dp_sgd, sweep_configuration_random_dp_sgd, \
    sweep_configuration_bayes_dp_sgd, sweep_configuration_bayes_logreg_dp

MICROBATCHES_STEP = 15


def sweep_train_dp_sgd():
    global MICROBATCHES_STEP

    wandb.init(
        # set the wandb project where this run will be logged
        project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning",

        # track hyperparameters and run metadata
        config=wandb.config
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

    model = dp_keras_model.DPSequential(
        l2_norm_clip=wandb.config.l2_norm_clip,
        noise_multiplier=wandb.config.noise_multiplier,
        num_microbatches=wandb.config.microbatches,  # wandb.config.microbatches
        layers=get_layers_Binary_Classification())
    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=wandb.config.epochs,
                        validation_data=(x_test, y_test),
                        batch_size=wandb.config.batch_size)
    end_time = time.time()

    epsilon = compute_epsilon_noise(wandb.config.epochs * x_train.shape[0] // wandb.config.batch_size,
                                    wandb.config.noise_multiplier,
                                    wandb.config.batch_size, total_samples)

    wandb.log({"Epsilon": epsilon})
    print("Epsilon: " + str(epsilon))

    training_time = round(end_time - start_time, 4)
    wandb.log({"Training_time": training_time})
    print("Training time: " + str(training_time))

    hist_df = pd.DataFrame(history.history)

    for index, epoch in hist_df.iterrows():
        print({'epochs': index,
               'loss': round(hist_df['loss'][index], 4),
               'acc': round(hist_df['accuracy'][index], 4),
               'val_loss': round(hist_df['val_loss'][index], 4),
               'val_acc': round(hist_df['val_accuracy'][index], 4)
               })
        wandb.log({'epochs': index,
                   'loss': round(hist_df['loss'][index], 4),
                   'accuracy': round(hist_df['accuracy'][index], 4),
                   'val_loss': round(hist_df['val_loss'][index], 4),
                   'val_accuracy': round(hist_df['val_accuracy'][index], 4)
                   })


def sweep_train_dp_logreg():
    wandb.init(
        # set the wandb project where this run will be logged
        project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning",

        # track hyperparameters and run metadata
        config=wandb.config
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

    start_time = time.time()

    train_dataset = RegressionDataset(points=x_train, labels=y_train, weights=None)
    test_dataset = RegressionDataset(points=x_test, labels=y_test, weights=None)

    epsilon = compute_epsilon_noise(wandb.config.epochs * x_train.shape[0] // wandb.config.batch_size,
                                    wandb.config.noise_multiplier,
                                    wandb.config.batch_size, total_samples)
    wandb.log({"Epsilon": epsilon})
    print("Epsilon: " + str(epsilon))

    # Bounds the probability of the privacy guarantee not holding. A rule of thumb is to set it to be less than
    # the inverse of the size of the training dataset. In this tutorial, it is set to 10^-6 as the dataset
    # has 180,000 training points.
    delta = 0.000001

    accuracies = logistic_dpsgd(train_dataset, test_dataset,
                                epsilon=epsilon,
                                delta=delta,
                                epochs=wandb.config.epochs,
                                num_classes=x_train.shape[0] + y_train.shape[0],
                                batch_size=wandb.config.batch_size,
                                num_microbatches=wandb.config.microbatches,
                                clipping_norm=wandb.config.l2_norm_clip
                                )

    end_time = time.time()

    training_time = round(end_time - start_time, 4)
    wandb.log({"Training_time": training_time})
    print("Training time: " + str(training_time))

    for index, accuracy in enumerate(accuracies):
        print({'epochs': index,
               'acc': round(accuracy, 4),
               })
        wandb.log({'epochs': index,
                   'accuracy': round(accuracy, 4)
                   })


def main():
    # wandb.login(key=api_key_wandb)

    sweep_id = wandb.sweep(sweep_configuration_grid_dp_sgd, project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                                                    "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_sgd(), count=147)

    sweep_id = wandb.sweep(sweep_configuration_random_dp_sgd,
                           project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_sgd(), count=100)

    sweep_id = wandb.sweep(sweep_configuration_bayes_dp_sgd,
                           project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_sgd(), count=100)

    sweep_id = wandb.sweep(sweep_configuration_bayes_logreg_dp, project="sweeps-hyperparameter-tuning-in-privacy"
                                                                        "-preserving"
                                                                        "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_logreg(), count=100)

    wandb.finish()


if __name__ == '__main__':
    main()
