
import pandas as pd
from IPython.display import display
import os


def main():
    df = pd.read_csv(os.environ['DATASET_PATH'])
    pd.set_option('display.max_columns', None)

    for column in df:
        print(df[column].value_counts())
    print(df.info())

    selected_rows = df.loc[[115419, 244054]]

    pivoted = selected_rows.transpose()

    print(pivoted.to_latex())


if __name__ == '__main__':
    main()
