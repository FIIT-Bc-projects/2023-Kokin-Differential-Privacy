import time

import pandas as pd
import wandb
import tensorflow as tf
from sklearn.model_selection import train_test_split

from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import get_layers_dnn, get_layers_logistic_regression
from scripts.sweep_configuration import sweep_configuration_bayes_baseline_logreg, \
    sweep_configuration_bayes_baseline_sgd_dnn, sweep_configuration_grid_baseline_logreg, \
    sweep_configuration_grid_baseline_sgd_dnn


def sweep_train_baseline_logreg():

    run = wandb.init(
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

    model_baseline_linear = tf.keras.Sequential(
        get_layers_logistic_regression())

    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)

    model_baseline_linear.compile(optimizer=optimizer,
                                  loss='mean_squared_error',
                                  metrics='accuracy')

    start_time = time.time()
    history = model_baseline_linear.fit(x_train, y_train,
                                        epochs=wandb.config.epochs,
                                        validation_data=(x_test, y_test),
                                        batch_size=wandb.config.batch_size)
    end_time = time.time()

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


def sweep_train_baseline_sgd():

    run = wandb.init(
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

    model_baseline_binary = tf.keras.Sequential(
        get_layers_dnn())

    optimizer = tf.keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)

    model_baseline_binary.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics='accuracy')

    start_time = time.time()
    history = model_baseline_binary.fit(x_train, y_train,
                                        epochs=wandb.config.epochs,
                                        validation_data=(x_test, y_test),
                                        batch_size=wandb.config.batch_size)
    end_time = time.time()

    training_time = str(round(end_time - start_time, 4))
    wandb.log({"Training_time": training_time})
    print("Training time: " + training_time)

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


def main():
    # wandb.login(key=api_key_wandb)

    sweep_id = wandb.sweep(sweep_configuration_grid_baseline_logreg,
                           project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_baseline_logreg, count=125)

    sweep_id = wandb.sweep(sweep_configuration_grid_baseline_sgd_dnn,
                           project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                   "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train_baseline_sgd, count=125)
    wandb.finish()


if __name__ == '__main__':
    main()
