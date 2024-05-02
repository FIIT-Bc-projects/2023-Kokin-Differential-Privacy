import datetime
import os
import time
from collections import Counter

import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
# import wandb
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.keras_models import dp_keras_model
import tensorflow as tf



from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import get_layers_dnn, get_layers_logistic_regression


def mia(name):
    learning_rate = 0.001
    epochs = 30
    batch_size = 5
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
    actual_dir = os.path.join("../Plots/MIA/" + name, timestamp)
    os.makedirs(actual_dir)

    noise_mul = 1
    microbatches = 1
    l2_norm_clip = 5

    total_num_of_models = 64
    selected_row_index = 30
    test_size = 0.2682926

    """ We need to have an array of models totally of total_num_of_models size"""
    models_with_sample_loss = []
    models_without_sample_loss = []

    """ Prepare and split data"""
    x, y = get_preprocessed_data()
    rows_to_keep = len(x) // 200
    x = x.iloc[:rows_to_keep]
    y = y[x.index]

    print(x.shape)
    print(y.shape)

    sample_x = x.loc[(selected_row_index,)]
    sample_y = y.loc[(selected_row_index,)]

    print(sample_x)

    for i in range(total_num_of_models):  # total_num_of_models/2 models with selected sample in training set and
                                          # other models without

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
        if i < (total_num_of_models / 2):  # first half of the models will be with the selected row
            while selected_row_index not in x_train.index:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
        else:  # other half will be without
            while selected_row_index in x_train.index:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)

        model = None
        if name == "DP_DNN":
            model = dp_keras_model.DPSequential(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_mul,
                num_microbatches=microbatches,  # wandb.config.microbatches
                layers=get_layers_dnn())
        elif name == "DP_LogReg":
            model = dp_keras_model.DPSequential(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_mul,
                num_microbatches=microbatches,  # wandb.config.microbatches
                layers=get_layers_logistic_regression())
        elif name == "BASELINE_DNN":
            model = keras.models.Sequential(
                layers=get_layers_dnn())
        elif name == "BASELINE_LogReg":
            model = keras.models.Sequential(
                layers=get_layers_logistic_regression())

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        start_time = time.time()
        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            batch_size=batch_size)
        end_time = time.time()

        training_time = round(end_time - start_time, 4)
        print("Training time: " + str(training_time))


        print("Iteration number " + str(i+1))
        print({'loss': round(history.history['loss'][-1], 4),
               'acc': round(history.history['accuracy'][-1], 4),
               'val_loss': round(history.history['val_loss'][-1], 4),
               'val_acc': round(history.history['val_accuracy'][-1], 4)
               })

        y_pred = model.predict(np.array([sample_x,]))
        print(y_pred)
        print(sample_y)
        ndigits = 2

        if name.startswith("BASELINE"):
            ndigits = 3

        mse = round(mean_squared_error(np.array([sample_y,]), y_pred), ndigits)
        if i < (total_num_of_models / 2):
            models_with_sample_loss.append(mse)
        else:
            models_without_sample_loss.append(mse)
    print("Name of experiment: " + name)
    print("Losses with sample in training: ")
    print(models_with_sample_loss)
    print("Losses without sample in training: ")
    print(models_without_sample_loss)


    # Bar plot
    # plt.figure(figsize=(8, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.grid(True)
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    ax1.hist(models_with_sample_loss, bins=15, color='blue', edgecolor='black')
    ax1.set_title('Frequency of Losses with Sample in training for ' + name)
    ax2.hist(models_without_sample_loss, bins=15, color='green', edgecolor='black')
    ax2.set_title('Frequency of Losses without Sample in training for ' + name)
    ax2.grid(True)
    plt.subplots_adjust(wspace=0.4,
                        hspace=0.4)
    plt.savefig(str(actual_dir) + "MIA_" + name, bbox_inches='tight')
    plt.show()


def main():
    mia("BASELINE_DNN")
    mia("BASELINE_LogReg")
    mia("DP_DNN")
    mia("DP_LogReg")


if __name__ == '__main__':
    main()
