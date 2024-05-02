import time
import numpy as np
import pandas as pd
import tensorflow_privacy
import wandb
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_privacy import logistic_dpsgd, DPSequential
from tensorflow_privacy.privacy.keras_models import dp_keras_model
from tensorflow_privacy.privacy.logistic_regression.datasets import RegressionDataset

from scripts.api_key_wandb import api_key_wandb
from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import compute_epsilon_noise, get_layers_logistic_regression, \
    get_layers_dnn
from scripts.sweep_configuration import sweep_configuration_bayes_logreg_dp, \
    sweep_configuration_grid_dp_logreg_patterns, \
    sweep_configuration_epsilon_grid, sweep_configuration_grid_dp_dnn_patterns, sweep_configuration_bayes_dnn_dp, \
    sweep_configuration_bayes_logreg_dp_learning_rate, sweep_configuration_bayes_dnn_dp_learning_rate, \
    sweep_configuration_grid_logreg_dp_microbatches, sweep_configuration_grid_dnn_dp_microbatches
from scripts.sweep_epsilon import sweep_grid_epsilon

MICROBATCHES_STEP = 15

wandb.login()


def sweep_train_dp_dnn():
    global MICROBATCHES_STEP

    wandb.init(
        # set the wandb project where this run will be logged
        project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning"
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
        layers=get_layers_dnn())
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

    wandb.log({"epsilon": epsilon})
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
        project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning"
    )
    """ Prepare and split data"""
    x, y = get_preprocessed_data()

    # Take a look at the number of rows
    # print("Data shape: " + str(x.shape))
    # print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    # print("X_train shape: " + str(x_train.shape))
    # print("X_test shape: " + str(x_test.shape))

    print("Shape of training set x: " + str(x_train.shape))
    print("Shape of training set y: " + str(y_train.shape))

    total_samples = x_train.shape[0]
    print("Total samples: ", total_samples)

    num_classes = 1  # y_train.shape[1] = 1
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]

    train_points_non_normalized = x_train.to_numpy().reshape(
        (num_train, -1))
    test_points_non_normalized = x_test.to_numpy().reshape(
        (num_test, -1))

    train_points = preprocessing.normalize(train_points_non_normalized)
    test_points = preprocessing.normalize(test_points_non_normalized)

    train_dataset = RegressionDataset(points=train_points, labels=y_train, weights=None)
    test_dataset = RegressionDataset(points=test_points, labels=y_test, weights=None)

    one_hot_test_labels = tf.one_hot(test_dataset.labels, num_classes)
    one_hot_train_labels = tf.one_hot(train_dataset.labels, num_classes)

    model = DPSequential(
        l2_norm_clip=wandb.config.l2_norm_clip,
        noise_multiplier=wandb.config.noise_multiplier,
        num_microbatches=wandb.config.microbatches,
        layers=get_layers_logistic_regression())

    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    start_time = time.time()

    history = model.fit(
        x_train,  # train_dataset.points,
        y_train,  # one_hot_train_labels,
        batch_size=wandb.config.batch_size,
        epochs=wandb.config.epochs,
        validation_data=(x_test, y_test)  # (test_dataset.points, one_hot_test_labels))
    )

    end_time = time.time()

    epsilon = compute_epsilon_noise(wandb.config.epochs * x_train.shape[0] // wandb.config.batch_size,
                                    wandb.config.noise_multiplier,
                                    wandb.config.batch_size, total_samples)

    wandb.log({"epsilon": epsilon})
    print("Epsilon: " + str(epsilon))

    training_time = round(end_time - start_time, 4)
    wandb.log({"Training_time": training_time})
    print("Training time: " + str(training_time))

    hist_df = pd.DataFrame(history.history)

    for index, epoch in hist_df.iterrows():
        print({'epochs': index + 1,
               'loss': round(hist_df['loss'][index], 4),
               'acc': round(hist_df['accuracy'][index], 4),
               'val_loss': round(hist_df['val_loss'][index], 4),
               'val_acc': round(hist_df['val_accuracy'][index], 4)
               })
        wandb.log({'epochs': index + 1,
                   'loss': round(hist_df['loss'][index], 4),
                   'accuracy': round(hist_df['accuracy'][index], 4),
                   'val_loss': round(hist_df['val_loss'][index], 4),
                   'val_accuracy': round(hist_df['val_accuracy'][index], 4)
                   })


def main():
    # wandb.login(key=api_key_wandb)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_grid_dp_sgd,
    #                        project="sweeps-hyperparameter-tuning-in-privacy-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_sgd, count=147)
    #
    # sweep_id = wandb.sweep(sweep=sweep_configuration_random_dp_sgd,
    #                        project="sweeps-hyperparameter-tuning-in-privacy-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_sgd, count=100)
    #
    # sweep_id = wandb.sweep(sweep=sweep_configuration_bayes_dnn_dp,
    #                        project="sweeps-hyperparameter-tuning-in-privacy-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_dnn, count=100)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_bayes_logreg_dp, project="sweeps-hyperparameter-tuning-in-privacy"
    #                                                                           "-preserving"
    #                                                                           "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_logreg, count=100)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_grid_dp_sgd_patterns, project="sweeps-hyperparameter-tuning-in"
    #                                                                                "-privacy"
    #                                                                                "-preserving"
    #                                                                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_sgd, count=200)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_epsilon_grid, project="sweeps-hyperparameter-tuning-in"
    #                                                                        "-privacy"
    #                                                                        "-preserving"
    #                                                                        "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_grid_epsilon, count=20)

    sweep_id = wandb.sweep(sweep=sweep_configuration_grid_dp_logreg_patterns, project="sweeps-hyperparameter-tuning-in"
                                                                                      "-privacy"
                                                                                      "-preserving"
                                                                                      "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_logreg, count=60)

    sweep_id = wandb.sweep(sweep=sweep_configuration_grid_dp_dnn_patterns, project="sweeps-hyperparameter-tuning-in"
                                                                                   "-privacy"
                                                                                   "-preserving"
                                                                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_dp_dnn, count=60)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_bayes_logreg_dp_learning_rate,
    #                        project="sweeps-hyperparameter-tuning-in-privacy"
    #                                "-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_logreg, count=100)
    #
    # sweep_id = wandb.sweep(sweep=sweep_configuration_bayes_dnn_dp_learning_rate,
    #                        project="sweeps-hyperparameter-tuning-in-privacy"
    #                                "-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_dnn, count=100)

    # sweep_id = wandb.sweep(sweep=sweep_configuration_grid_logreg_dp_microbatches,
    #                        project="sweeps-hyperparameter-tuning-in-privacy"
    #                                "-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_logreg, count=32)
    #
    # sweep_id = wandb.sweep(sweep=sweep_configuration_grid_dnn_dp_microbatches,
    #                        project="sweeps-hyperparameter-tuning-in-privacy"
    #                                "-preserving"
    #                                "-machine-learning")
    # wandb.agent(sweep_id, function=sweep_train_dp_dnn, count=32)

    wandb.finish()


if __name__ == '__main__':
    main()
