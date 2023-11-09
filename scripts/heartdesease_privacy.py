""" IMPORTING LIBRARIES AND LOADING DATA """
import datetime
import os
import sys
import dp_accounting
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import missingno as msno
import tensorflow_privacy
from absl import flags
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow_privacy import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
from tensorflow_privacy.privacy.keras_models import dp_keras_model
import wandb
from environs import Env

""" Environmental variables """
env = Env()

""" Directories and timestamps """
actual_dir = None
timestamp = None
logger = None
""" Hyperparameters """

flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1, 'Clipping norm')
flags.DEFINE_integer('batch_size', 50, 'Batch size')
flags.DEFINE_integer('epochs', 6, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 50, 'Number of microbatches '
                        '(must evenly divide batch_size)')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

""" Create and compile model"""


def get_layers_Binary_Classification():
    return [tf.keras.layers.InputLayer(input_shape=(39,)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')]


def get_layers_linear_regression():
    return [tf.keras.layers.Dense(1, activation="linear")]


def create_baseline_models():
    model = []

    """Regular Binary Classification Baseline"""
    model_baseline_binary = tf.keras.Sequential(
        get_layers_Binary_Classification())

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    model_baseline_binary.compile(optimizer=optimizer,
                                  loss='mse',
                                  metrics='accuracy')

    model_baseline_linear = tf.keras.Sequential(
        get_layers_Binary_Classification())

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    model_baseline_linear.compile(optimizer=optimizer,
                                  loss='mean_squared_error',
                                  metrics='accuracy')

    model.append(model_baseline_binary)
    model.append(model_baseline_linear)
    return model


""" Computes epsilon value for given hyperparameters """


def compute_epsilon(steps):
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = FLAGS.batch_size / 180000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 239846 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def compute_epsilon_noise(steps, noise_multiplier):
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = FLAGS.batch_size / 180000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 239846 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def get_preprocessed_data():
    df = pd.read_csv(os.environ['DATASET_PATH'])

    logger.log({"dataset": wandb.Table(dataframe=df)})

    print(df.LastCheckupTime.value_counts())
    print(df.State.value_counts())
    print(df.HadHeartAttack.unique)

    y = df.HadHeartAttack.replace(['Yes', 'No'], [1, 0])

    x = df.drop('HadHeartAttack', axis=1)
    pd.set_option('display.max_columns', None)
    display(x)

    # encode categorical data to numerical
    enc = LabelEncoder()
    for i in x.columns:
        if x[i].dtype == 'object':
            x[i] = enc.fit_transform(x[i])
    print(x.info())

    x.drop(axis=0, index=x.index[-22:], inplace=True)
    y.drop(axis=0, index=y.index[-22:], inplace=True)

    return x, y


""" Grid search approach"""


def calculate_model(x_train, x_test, y_train, y_test, m, index):
    if FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')

    model = dp_keras_model.DPSequential(
        l2_norm_clip=m["l2_norm_clip"],
        noise_multiplier=m["noise_multiplier"],
        num_microbatches=FLAGS.microbatches,
        layers=get_layers_Binary_Classification())

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    # optimizer = DPKerasSGDOptimizer(
    #                 l2_norm_clip=m["l2_norm_clip"],
    #                 noise_multiplier=m["noise_multiplier"],
    #                 num_microbatches=FLAGS.microbatches,
    #                 learning_rate=FLAGS.learning_rate)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    print("Training with batch_size = " + str(FLAGS.batch_size) +
          ", microbatches = " + str(FLAGS.microbatches) +
          ", l2_norm_clip = " + str(m["l2_norm_clip"]) +
          ", noise_multiplier = " + str(m["noise_multiplier"]) +
          ", learning_rate = " + str(FLAGS.learning_rate))

    history = model.fit(x_train, y_train,
                        epochs=FLAGS.epochs,
                        validation_data=(x_test, y_test),
                        batch_size=FLAGS.batch_size)

    m['acc'] = round(history.history['accuracy'][-1], 4)
    m['val_acc'] = round(history.history['val_accuracy'][-1], 4)
    m['loss'] = round(history.history['loss'][-1], 4)
    m['val_loss'] = round(history.history['val_loss'][-1], 4)

    print("Best metrics are: acc = " + str(history.history['accuracy'][-1]) +
          ", val_acc = " + str(history.history['val_accuracy'][-1]) +
          ", loss = " + str(history.history['loss'][-1]) +
          ", val_loss = " + str(history.history['val_loss'][-1]))

    log_plots_dp(history, m, index)


def log_plots_dp(history, m, index):
    history_dict = history.history
    history_df = pd.DataFrame(history_dict)
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title(
    #     'DP Model accuracy with noise: ' + str(m['noise_multiplier']) + ', l2_norm_clip: ' + str(m['l2_norm_clip']))
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.grid(True)
    # plt.tight_layout()
    # # save png of the plot
    # plt.savefig(str(actual_dir) + "dp_acc_n_" + str(m['noise_multiplier']).replace(".", "_")
    #             + "l2_" + str(m['l2_norm_clip']).replace(".", "_"), bbox_inches='tight')
    #
    # plt.show()

    # summarize history for loss
    plt.plot(history.history['accuracy'])
    for i, metric in enumerate(history.history['accuracy']):
        plt.text(i, history.history['accuracy'][i], f'{round(metric, 4)}', ha='right')

    plt.plot(history.history['val_accuracy'])
    for i, metric in enumerate(history.history['val_accuracy']):
        plt.text(i, history.history['val_accuracy'][i], f'{round(metric, 4)}', ha='left')

    plt.plot(history.history['loss'])
    for i, metric in enumerate(history.history['loss']):
        plt.text(i, history.history['loss'][i], f'{round(metric, 4)}', ha='left')

    plt.plot(history.history['val_loss'])
    for i, metric in enumerate(history.history['val_loss']):
        plt.text(i, history.history['val_loss'][i], f'{round(metric, 4)}', ha='right')

    plt.title(
        'DP Model acc and loss with noise: ' + str(m['noise_multiplier']) + ', l2_norm_clip: ' + str(m['l2_norm_clip']))
    plt.ylabel('loss, acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # save png of the plot
    plt.savefig(str(actual_dir) + "dp_n_" + str(m['noise_multiplier']).replace(".", "_")
                + "l2_" + str(m['l2_norm_clip']).replace(".", "_"), bbox_inches='tight')
    plt.show()

    title = "Accuracy and Loss metrics of DP model training: noise=%s, l2_norm_clip=%s" % (str(m['noise_multiplier']),
                                                                                           str(m['l2_norm_clip']))

    wandb.log(
        {"Accuracy and Loss metrics of DP model (" + str(index) + ")":
            wandb.plot.line_series(
                xs=history_df.index,
                ys=[history_df["loss"],
                    history_df["val_loss"],
                    history_df["accuracy"],
                    history_df["val_accuracy"]
                    ],
                keys=["loss", "val_loss", "accuracy", "val_accuracy"],
                title=title,
                xname="epochs")})

    # log wandb plot
    # title = "Accuracy metrics of DP model training: noise=%s, l2_norm_clip=%s" % (str(m['noise_multiplier']),
    #                                                                               str(m['l2_norm_clip']))
    # wandb.log(
    #     {"Training accuracy metrics of DP model (" + str(index) + ")":
    #         wandb.plot.line_series(
    #             xs=history_df.index,
    #             ys=[history_df["accuracy"],
    #                 history_df["val_accuracy"]],
    #             keys=["accuracy", "val_accuracy"],
    #             title=title,
    #             xname="epochs")})
    # title = "Loss metrics of DP model training: noise=%s, l2_norm_clip=%s" % (str(m['noise_multiplier']),
    #                                                                           str(m['l2_norm_clip']))
    # wandb.log(
    #     {"Training loss metrics of DP model (" + str(index) + ")":
    #         wandb.plot.line_series(
    #             xs=history_df.index,
    #             ys=[history_df["loss"],
    #                 history_df["val_loss"]],
    #             keys=["loss", "val_loss"],
    #             title=title,
    #             xname="epochs")})


def log_plots_baseline(history, index):
    history_dict = history.history
    history_df = pd.DataFrame(history_dict)
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    for i, metric in enumerate(history.history['accuracy']):
        plt.text(i, history.history['accuracy'][i], f'{round(metric, 4)}', ha='left')

    plt.plot(history.history['val_accuracy'])
    for i, metric in enumerate(history.history['val_accuracy']):
        plt.text(i, history.history['val_accuracy'][i], f'{round(metric, 4)}', ha='right')

    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if index == 0:
        plt.title('Binary model accuracy metrics with batch_size = ' + str(FLAGS.batch_size) + ', microbatches = '
                  + str(FLAGS.microbatches) + ", ")

        plt.savefig(str(actual_dir) + "baseline_binary_acc", bbox_inches='tight')
        wandb.log(
            {"Accuracy metrics of baseline Binary Classification model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["accuracy"],
                        history_df["val_accuracy"]],
                    keys=["accuracy", "val_accuracy"],
                    title="Accuracy metrics of baseline Binary Classification model",
                    xname="epochs"
                )})
    else:
        plt.title('Linear model accuracy metrics with batch_size = ' + str(FLAGS.batch_size) + ', microbatches = '
                  + str(FLAGS.microbatches) + ", ")

        plt.savefig(str(actual_dir) + "baseline_linear_acc", bbox_inches='tight')
        wandb.log(
            {"Loss metrics of baseline Binary Classification model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["accuracy"],
                        history_df["val_accuracy"]],
                    keys=["accuracy", "val_accuracy"],
                    title="Accuracy metrics of baseline Linear regression model",
                    xname="epochs")})
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    for i, metric in enumerate(history.history['loss']):
        plt.text(i, history.history['loss'][i], f'{round(metric, 4)}', ha='left')

    plt.plot(history.history['val_loss'])
    for i, metric in enumerate(history.history['val_loss']):
        plt.text(i, history.history['val_loss'][i], f'{round(metric, 4)}', ha='right')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if index == 0:
        title = 'Binary model loss metrics with batch_size = ' + str(FLAGS.batch_size) + ', microbatches = ' \
                + str(FLAGS.microbatches) + ", "
        plt.title(title)

        plt.savefig(str(actual_dir) + "baseline_binary_loss", bbox_inches='tight')
        wandb.log(
            {"Loss metrics of baseline Binary Classification model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["loss"],
                        history_df["val_loss"]],
                    keys=["loss", "val_loss"],
                    title="Loss metrics of baseline Binary Classification model",
                    xname="epochs")})

    else:
        title = 'Linear model loss metrics with batch_size = ' + str(FLAGS.batch_size) + ', microbatches = ' \
                + str(FLAGS.microbatches) + ", "
        plt.title(title)
        plt.savefig(str(actual_dir) + "baseline_linear_loss", bbox_inches='tight')
        wandb.log(
            {"Loss metrics of baseline Linear Regression model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["loss"],
                        history_df["val_loss"]],
                    keys=["loss", "val_loss"],
                    title="Loss metrics of baseline Linear Regression model",
                    xname="epochs")})
    plt.show()


def main():
    global timestamp, actual_dir, logger
    """ Initialize wandb"""
    logger = wandb.init(
        # set the wandb project where this run will be logged
        project="Hyperparameter Tuning in Privacy-Preserving Machine Learning",

        # track hyperparameters and run metadata
        config={
            "learning_rate": FLAGS.learning_rate,
            "dataset": "heart_2022_no_nans",
            "epochs": FLAGS.epochs,
        }
    )

    """ Create directories for plots """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
    actual_dir = os.path.join("../Plots/", timestamp)
    os.makedirs(actual_dir)

    """ Prepare and split data"""
    x, y = get_preprocessed_data()
    # Take a look at the number of rows
    print("Data shape: " + str(x.shape))
    print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    print("X_train shape: " + str(x_train.shape))
    print("X_test shape: " + str(x_test.shape))

    """Get baseline results"""
    models = create_baseline_models()
    for index, model in enumerate(models):
        history = model.fit(x_train, y_train,
                            epochs=FLAGS.epochs,
                            validation_data=(x_test, y_test),
                            batch_size=FLAGS.batch_size)

        log_plots_baseline(history, index)
    """ Grid Search """
    model_grid = []

    aVals = [.25, .5, 1, 1.25, 1.5, 2]

    for norm_clip in aVals:
        for noise_mul in aVals:
            model_grid.append({
                'l2_norm_clip': FLAGS.l2_norm_clip * norm_clip,
                'noise_multiplier': FLAGS.noise_multiplier * noise_mul,
                'epochs': FLAGS.epochs,
                'acc': 0,
                'val_acc': 0,
                'loss': 0,
                'val_loss': 0
            })

    for i, m in enumerate(model_grid):
        calculate_model(
            x_train, x_test, y_train, y_test, m, i
        )

    # calculate epsilon
    if env.bool("DP", False) is True:
        privacy_results = pd.DataFrame(columns=['epsilon', 'noise_multiplier', 'l2_norm_clip',
                                                'acc', 'val_acc', 'loss', 'val_loss'])
        for m in model_grid:
            epsilon = compute_epsilon_noise(FLAGS.epochs * 180000 // FLAGS.batch_size, m['noise_multiplier'])
            new_row = {'epsilon': epsilon,
                       'noise_multiplier': m['noise_multiplier'],
                       'l2_norm_clip': m['l2_norm_clip'],
                       'acc': m['acc'],
                       'val_acc': m['val_acc'],
                       'loss': m['loss'],
                       'val_loss': m['val_loss']
                       }
            privacy_results.loc[len(privacy_results)] = new_row
            print("Computed epsilon with l2_norm_clip = " + str(m['l2_norm_clip']) + ", noise_multiplier = " +
                  str(m['noise_multiplier']) + " is epsilon = " + str(epsilon))

        uniques = privacy_results['l2_norm_clip'].unique()
        print(uniques)
        for u in uniques:
            actual = privacy_results[privacy_results["l2_norm_clip"] == u]
            # sorted_actual = sorted(actual, key=lambda x: x["val_acc"], reverse=True)
            print(actual)
            # plot epsilon graph
            plt.plot(actual["noise_multiplier"], actual["epsilon"])
            plt.title('model epsilon values for l2_norm_clip ' + str(u))
            plt.ylabel('epsilon')
            plt.xlabel('noise_multiplier')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(actual_dir + "epsilon_l2_norm_clip_" + str(u).replace(".", '_'), bbox_inches='tight')
            logger.log({"Epsilon plot l2 = " + str(u): plt})
            plt.show()
            # plot accuracy graph
            plt.plot(actual["noise_multiplier"], actual["val_acc"])
            plt.title('model val_acc values for l2_norm_clip ' + str(u))
            plt.ylabel('val_acc')
            plt.xlabel('noise_multiplier')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(actual_dir + "noise_accuracy_l2_norm_clip_" + str(u).replace(".", '_'), bbox_inches='tight')
            logger.log({"Accuracy by noise and clip plot l2 = " + str(u): plt})
            plt.show()

    wandb.finish()


if __name__ == '__main__':
    main()
