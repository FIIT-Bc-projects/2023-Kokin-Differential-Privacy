import os

import pandas as pd
from IPython.core.display_functions import display
from sklearn.preprocessing import LabelEncoder


def get_preprocessed_data(logger, wandb):
    df = pd.read_csv(os.environ['DATASET_PATH'])

    logger.log({"dataset": wandb.Table(dataframe=df)})

    # print(df.LastCheckupTime.value_counts())
    # print(df.State.value_counts())
    # print(df.HadHeartAttack.unique)

    y = df.HadHeartAttack.replace(['Yes', 'No'], [1, 0])

    x = df.drop('HadHeartAttack', axis=1)
    pd.set_option('display.max_columns', None)
    # display(x)

    # encode categorical data to numerical
    enc = LabelEncoder()
    for i in x.columns:
        if x[i].dtype == 'object':
            x[i] = enc.fit_transform(x[i])
    print(x.info())

    x.drop(axis=0, index=x.index[-22:], inplace=True)
    y.drop(axis=0, index=y.index[-22:], inplace=True)

    return x, y



