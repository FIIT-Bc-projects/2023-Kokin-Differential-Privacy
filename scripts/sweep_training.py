import time

import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_privacy.privacy.keras_models import dp_keras_model

from scripts.api_key_wandb import api_key_wandb
from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import get_layers_Binary_Classification
from scripts.sweep_configuration import sweep_configuration_bayes


logger = None


def sweep_train():
    global logger
    """ Prepare and split data"""
    x, y = get_preprocessed_data(logger, wandb)
    # Take a look at the number of rows
    # print("Data shape: " + str(x.shape))
    # print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    # print("X_train shape: " + str(x_train.shape))
    # print("X_test shape: " + str(x_test.shape))

    total_samples = x_train.shape[0]
    total_steps = int(np.ceil(total_samples / wandb.config.batch_size)) * wandb.config.epochs
    steps_per_epoch = int(np.ceil(total_steps / wandb.config.epochs))

    model = dp_keras_model.DPSequential(
        l2_norm_clip=wandb.config.l2_norm_clip,
        noise_multiplier=wandb.config.noise_multiplier,
        num_microbatches=wandb.config.microbatches,
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

    print("Training time: " + str(round(end_time - start_time, 4)))

    hist_df = pd.DataFrame(history.history)

    wandb.log_artifact(model)
    for index, epoch in hist_df.iterrows():
        wandb.log({'epochs': index,
                   'loss': round(hist_df['loss'][index], 4),
                   'acc': round(hist_df['accuracy'][index], 4),
                   'val_loss': round(hist_df['val_loss'][index], 4),
                   'val_acc': round(hist_df['val_accuracy'][index], 4)
                   })


def main():
    global logger
    wandb.login(key=api_key_wandb)
    logger = wandb.init(
        # set the wandb project where this run will be logged
        project="Hyperparameter Tuning in Privacy-Preserving Machine Learning",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.00001,
            "dataset": "heart_2022_no_nans",
            "epochs": 6,
        }
    )
    sweep_id = wandb.sweep(sweep_configuration_bayes, project="sweeps-hyperparameter-tuning-in-privacy-preserving"
                                                              "-machine-learning")
    wandb.agent(sweep_id, function=sweep_train, count=10)
    wandb.finish()

if __name__ == '__main__':
    main()
