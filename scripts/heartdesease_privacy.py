# IMPORTING LIBRARIES AND LOADING DATA
import sys

import dp_accounting
# Commented out IPython magic to ensure Python compatibility.
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
from tensorflow_privacy.privacy.keras_models import dp_keras_model
import wandb

"""## Hyperparameters"""


flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
                   'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.015, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.5,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.5, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 1, 'Number of microbatches '
                       '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS
FLAGS(sys.argv)
wandb.init(config=FLAGS, sync_tensorboard=True)


def create_model():
    model = dp_keras_model.DPSequential(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        layers=[
            tf.keras.layers.InputLayer(input_shape=(17,)),
            tf.keras.layers.Dense(
                51, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(81, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])

    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


def compute_epsilon(steps):
    """Computes epsilon value for given hyperparameters."""
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = FLAGS.batch_size / 239846
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(FLAGS.noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 239846 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def get_preprocessed_data():
    df = pd.read_csv('D:/Study/BP/Implementation/Dataset/heart_2020_cleaned.csv')

    """# EXPLORATORY DATA ANALYSIS"""
    # msno.matrix(df)
    # plt.figure(figsize = (15,9))
    # plt.show()

    df.HeartDisease.value_counts()

    # plt.figure(figsize=(10,6))
    # sns.countplot(x='HeartDisease', data=df).set_title('Distribution of Target Variable')

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

    # plt.figure(figsize=(16, 8))
    # for i in cat:
    #     sns.catplot(y='HeartDisease', x=i, hue='Sex', data=df, kind='bar')
    #
    # for i in cat:
    #     sns.catplot(y='HeartDisease', x=i, hue='Sex', data=df, kind='point')

    allc = [col for col in df.columns]
    imp = list(set(allc) - set(cat))

    si = LabelEncoder()
    for i in cat:
        si.fit(df[i])
        df[i] = si.transform(df[i])

    return df


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="Hyperparameter Tuning in Privacy-Preserving Machine Learning",

        # track hyperparameters and run metadata
        config={
            "learning_rate": FLAGS.learning_rate,
            "architecture": "CNN",
            "dataset": "heart_2020_cleared",
            "epochs": FLAGS.epochs,
        }
    )

    df = get_preprocessed_data()

    """# Training with DP"""

    allcol = [col for col in df.columns]
    allcol.remove('HeartDisease')

    x = df[allcol]
    y = df['HeartDisease']
    df['AgeCategory'] = df['AgeCategory'] / 80

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    """Classifier wandb"""
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

    """Grid Search"""
    # model = KerasClassifier(model=create_model, verbose=0)
    # batch_size = [50, 150, 250]
    # epochs = [30]
    # param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    # grid_result = grid.fit(x_train, y_train)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    """No Grid Search"""
    model = create_model()
    history = model.fit(x_train, y_train,
                        epochs=FLAGS.epochs,
                        validation_data=(x_test, y_test),
                        batch_size=FLAGS.batch_size)
    print("History keys are: " + str(history.history.keys()))
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

    print('Computed epsilon for delta=1e-6: %.2f' % compute_epsilon(FLAGS.epochs * 239846 // FLAGS.batch_size))


if __name__ == '__main__':
    main()




