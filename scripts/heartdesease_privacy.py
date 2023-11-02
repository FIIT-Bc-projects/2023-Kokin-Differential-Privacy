""" IMPORTING LIBRARIES AND LOADING DATA """
import os
import sys
import dp_accounting
import pandas as pd
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

""" Hyperparameters """

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                   'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.5, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 3, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 256, 'Number of microbatches '
                         '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

# wandb.init(config=FLAGS, sync_tensorboard=True)

""" Create and compile model"""


def create_model():
    if env.bool("DP", False) is True:

        """DP model with regular optimizer and regular loss"""
        model = dp_keras_model.DPSequential(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            layers=[
                tf.keras.layers.InputLayer(input_shape=(17,)),
                tf.keras.layers.Dense(51, activation='relu'),
                tf.keras.layers.Dense(81, activation='relu'),
                tf.keras.layers.Dense(10, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')])

        optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=['accuracy']
        )

        return model
    else:
        """Regular model """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(17,)),
            tf.keras.layers.Dense(51, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics='accuracy')
        return model

    # TODO append a tf_privacy Estimator with Differentially private model, optimizer and loss function,
    #  implement main


""" Computes epsilon value for given hyperparameters """


def compute_epsilon(steps):
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = FLAGS.batch_size / 256000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 239846 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def compute_epsilon(steps, noise_multiplier):
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = FLAGS.batch_size / 256000
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 239846 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def get_preprocessed_data():
    df = pd.read_csv(os.environ['DATASET_PATH'])

    """ Exploratory data analysis and preprocessing"""
    if env.bool("EDA_DETAILED", False) is True:
        msno.matrix(df)
        plt.figure(figsize=(15, 9))
        plt.show()

    df.HeartDisease.value_counts()

    if env.bool("EDA_DETAILED", False) is True:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='HeartDisease', data=df).set_title('Distribution of Target Variable')

    df['Diabetic'].unique()

    diabetic = {'Yes': 0.80, 'No': 0.00, 'No, borderline diabetes': 0.5, 'Yes (during pregnancy)': 1.00}
    df['Diabetic'] = df['Diabetic'].apply(lambda ld: diabetic[ld])
    df['Diabetic'] = df['Diabetic'].astype('float')

    df['AgeCategory'].unique()
    ageCategory = {'55-59': 57, '80 or older': 80, '65-69': 67, '75-79': 77, '40-44': 42, '70-74': 72,
                   '60-64': 62, '50-54': 52, '45-49': 47, '18-24': 21, '35-39': 37, '30-34': 32, '25-29': 27}
    df['AgeCategory'] = df['AgeCategory'].apply(lambda ld: ageCategory[ld])
    df['AgeCategory'] = df['AgeCategory'].astype('float')

    df['Race'].unique()

    si = LabelEncoder()
    si.fit(df['HeartDisease'])
    df['HeartDisease'] = si.transform(df['HeartDisease'])

    df.head()

    cat = [col for col in df.columns if df[col].dtypes == 'object']

    if env.bool("EDA_DETAILED", False) is True:
        plt.figure(figsize=(16, 8))
        for i in cat:
            sns.catplot(y='HeartDisease', x=i, hue='Sex', data=df, kind='bar')

        for i in cat:
            sns.catplot(y='HeartDisease', x=i, hue='Sex', data=df, kind='point')

    allc = [col for col in df.columns]
    print(list(set(allc) - set(cat)))

    si = LabelEncoder()
    for i in cat:
        si.fit(df[i])
        df[i] = si.transform(df[i])

    return df


""" Grid search approach"""


def calculate_model(x_train, x_test, y_train, y_test, m):

    if FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')

    model = dp_keras_model.DPSequential(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(17,)),
            tf.keras.layers.Dense(51, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

    # optimizer = DPKerasSGDOptimizer(
    #     l2_norm_clip=m['l2_norm_clip'],
    #     noise_multiplier=m['noise_multiplier'],
    #     num_microbatches=FLAGS.microbatches,
    #     learning_rate=FLAGS.learning_rate)
    #
    # loss = tf.keras.losses.BinaryCrossentropy(
    #     from_logits=True, reduction=tf.losses.Reduction.NONE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)

    model.compile(optimizer=optimizer, loss="mse", metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        epochs=FLAGS.epochs,
                        validation_data=(x_test, y_test),
                        batch_size=FLAGS.batch_size)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy with noise: ' + str(m['noise_multiplier']) + ', l2_norm_clip: ' + str(m['l2_norm_clip']))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss with noise: ' + str(m['noise_multiplier']) + ', l2_norm_clip: ' + str(m['l2_norm_clip']))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def main():
    """ Initialize wandb"""
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Hyperparameter Tuning in Privacy-Preserving Machine Learning",
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": FLAGS.learning_rate,
    #         "architecture": "CNN",
    #         "dataset": "heart_2020_cleared",
    #         "epochs": FLAGS.epochs,
    #     }
    # )

    """ Prepare and split data"""
    df = get_preprocessed_data()
    # Take a look at the number of rows
    print(df.shape)

    allcol = [col for col in df.columns]
    allcol.remove('HeartDisease')

    x = df[allcol]
    y = df['HeartDisease']
    df['AgeCategory'] = df['AgeCategory'] / 80

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.19948717)
    print(x_train.shape)
    print(x_test.shape)

    """ Classifier wandb """
    # # Define the training inputs
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": input(x_train)[0]},
    #     y=input(y_train)[1],
    #     num_epochs=FLAGS.epochs,
    #     batch_size=FLAGS.batch_size,
    #     shuffle=True,
    # )
    #
    # # NOTE: We change the summary logging frequency to be every epoch with save_summary_steps
    # classifier = tf.estimator.DNNClassifier(
    #     feature_columns=[tf.feature_column.numeric_column("x", shape=[17])],
    #     hidden_units=[256, 32],
    #     optimizer=tf.train.AdamOptimizer(1e-6),
    #     n_classes=10,
    #     dropout=0.1,
    #     config=tf.estimator.RunConfig(
    #         save_summary_steps=x_train.shape[0] / wandb.config.batch_size)
    # )
    # # Train the classifier
    # classifier.train(input_fn=train_input_fn, steps=wandb.config.max_steps)
    #
    # # Define the test inputs
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": input(x_test)[0]},
    #     y=input(y_test)[1],
    #     num_epochs=1,
    #     shuffle=False
    # )
    #
    # # Evaluate accuracy
    # accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    # print(f"\nTest Accuracy: {accuracy_score:.0%}\n")

    """ Grid Search """
    if env.bool("GRID", False) is True:
        model_grid = []

        aVals = [1 / 8, .25, .5, 1, 2]

        for norm_clip in aVals:
            for noise_mul in aVals:
                model_grid.append({
                    'l2_norm_clip': FLAGS.l2_norm_clip * norm_clip,
                    'noise_multiplier': FLAGS.noise_multiplier * noise_mul,
                    'epochs': FLAGS.epochs,
                })

        for m in model_grid:
            calculate_model(
                x_train, x_test, y_train, y_test, m
            )

        # calculate epsilon
        if env.bool("DP", False) is True:
            privacy_results = pd.DataFrame(columns=['epsilon', 'noise_multiplier', 'l2_norm_clip'])
            for m in model_grid:
                epsilon = compute_epsilon(FLAGS.epochs * 256000 // FLAGS.batch_size, m['noise_multiplier'])
                new_row = {'epsilon': epsilon, 'noise_multiplier': m['noise_multiplier'], 'l2_norm_clip': m['l2_norm_clip']}
                privacy_results.concat(new_row, inplace=True)
                print('Computed epsilon for delta=1e-6: %.2f' % epsilon)

            uniques = privacy_results['l2_norm_clip'].unique()
            for u in uniques:
                actual = privacy_results[privacy_results["l2_norm_clip"] == u]
                plt.plot(actual)
                plt.title('model epsilon values for l2_norm_clip ' + str(actual['l2_norm_clip']))
                plt.ylabel('epsilon')
                plt.xlabel('noise_multiplier')
                plt.show()

    else:
        """No Grid Search"""
        model = create_model()
        history = model.fit(x_train, y_train,
                            epochs=FLAGS.epochs,
                            validation_data=(x_test, y_test),
                            batch_size=FLAGS.batch_size)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        if env.bool("DP", False) is True:
            print('Computed epsilon for delta=1e-6: %.2f' % compute_epsilon(FLAGS.epochs * 256000 // FLAGS.batch_size))


if __name__ == '__main__':
    main()
