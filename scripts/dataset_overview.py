
import pandas as pd
from IPython.display import display
from tabulate import tabulate
import os


def main():
    df = pd.read_csv(os.environ['DATASET_PATH'])
    pd.set_option('display.max_columns', None)
    print(tabulate(df.loc[[200, 10456, 74345, 115423, 142144, 159220, 160435, 174254, 198453, 212345, 234796, 244054]],
                   headers='keys', tablefmt='psql'))

    for column in df:
        print(df[column].value_counts())
    print(df.info())

    print(df.loc[[200, 10456, 74345, 115423, 142144, 159220, 160435, 174254, 198453, 212345, 234796, 244054]].to_latex())


if __name__ == '__main__':
    main()