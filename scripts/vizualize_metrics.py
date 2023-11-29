import os

import openpyxl
import pandas as pd
import matplotlib.pyplot as plt


def main():
    metrics_avgs_dir = '../Metrics/Metrics_Averages/'
    if not os.path.exists(metrics_avgs_dir):
        os.makedirs(metrics_avgs_dir)

    # Specify the path to your Excel file
    excel_in_file_path = '../Metrics/metrics.xlsx'
    excel_dir = os.path.join("../", "Metrics/")
    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)
    # setup excel_path
    excel_path = os.path.join(metrics_avgs_dir, "metrics_avgs.xlsx")
    book = openpyxl.load_workbook(excel_path)
    writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay')
    writer.book = book

    # Get the list of all sheet names in the Excel file
    all_sheet_names = pd.ExcelFile(excel_in_file_path).sheet_names

    # Specify the columns you want to read (if needed)
    columns_to_read = ['model', 'optimizer', 'epsilon', 'noise_multiplier', 'l2_norm_clip', 'batch_size',
                       'microbatches', 'learning_rate', 'training_time', 'acc', 'val_acc', 'loss', 'val_loss']

    columns_to_average = ['training_time', 'acc', 'val_acc', 'loss', 'val_loss']
    res_columns = ['model', 'optimizer', 'epsilon', 'noise_multiplier', 'l2_norm_clip', 'batch_size',
                   'microbatches', 'learning_rate',
                   'avg_training_time', 'avg_acc', 'avg_val_acc', 'avg_loss', 'avg_val_loss']

    df_res = pd.DataFrame(columns=res_columns)

    for sheet_name in all_sheet_names:
        if sheet_name == 'Baseline':
            continue
        df = pd.read_excel(excel_in_file_path, sheet_name=sheet_name)
        unique_noises = df['noise_multiplier'].unique()
        unique_clips = df['l2_norm_clip'].unique()
        excel_sheet_dir = os.path.join(excel_dir, sheet_name + "Plots/")
        if not os.path.exists(excel_sheet_dir):
            os.makedirs(excel_sheet_dir)
        for unique_clip in unique_clips:
            df_noise = pd.DataFrame(columns=res_columns)
            for unique_noise in unique_noises:
                actual_rows = df[(df['noise_multiplier'] == unique_noise) &
                                 (df['l2_norm_clip'] == unique_clip)]

                num_rows = len(actual_rows)

                new_row = {'model': actual_rows['model'][0],
                           'optimizer': actual_rows['optimizer'][0],
                           'epsilon': actual_rows['epsilon'][0],
                           'noise_multiplier': actual_rows['noise_multiplier'][0],
                           'l2_norm_clip': actual_rows['l2_norm_clip'][0],
                           'batch_size': actual_rows['batch_size'][0],
                           'microbatches': actual_rows['microbatches'][0],
                           'learning_rate': actual_rows['learning_rate'],
                           'avg_training_time': sum(actual_rows['training_time']) / num_rows,
                           'avg_acc': sum(actual_rows['acc']) / num_rows,
                           'avg_val_acc': sum(actual_rows['val_acc']) / num_rows,
                           'avg_loss': sum(actual_rows['loss']) / num_rows,
                           'avg_val_loss': sum(actual_rows['val_loss']) / num_rows,
                           }

                df_noise.loc[len(df_res)] = new_row
                df_res.loc[len(df_res)] = new_row

            # Plot the averages for hyperparams
            plt.plot(df_noise['noise_multiplier'], df_noise['avg_acc'])
            for i, metric in enumerate(df_noise['avg_acc']):
                plt.text(i, df_noise['avg_acc'][i], f'{round(metric, 4)}', ha='right')

            plt.plot(df_noise['noise_multiplier'], df_noise['avg_val_acc'])
            for i, metric in enumerate(df_noise['avg_val_acc']):
                plt.text(i, df_noise['avg_val_acc'][i], f'{round(metric, 4)}', ha='right')

            plt.plot(df_noise['noise_multiplier'], df_noise['avg_loss'])
            for i, metric in enumerate(df_noise['avg_loss']):
                plt.text(i, df_noise['avg_loss'][i], f'{round(metric, 4)}', ha='right')

            plt.plot(df_noise['noise_multiplier'], df_noise['avg_val_loss'])
            for i, metric in enumerate(df_noise['avg_val_loss']):
                plt.text(i, df_noise['avg_val_loss'][i], f'{round(metric, 4)}', ha='right')

            plt.plot(df_noise['noise_multiplier'], df_noise['avg_training_time'])
            for i, metric in enumerate(df_noise['avg_training_time']):
                plt.text(i, df_noise['avg_training_time'][i], f'{round(metric, 4)}', ha='right')

            plt.title(
                'DP Model (' + df_noise['optimizer'][0] + ') averages with l2_norm_clip: ' +
                str(df_noise['l2_norm_clip'][0]))
            plt.ylabel('loss, acc, time,')
            plt.xlabel('noise_multiplier')
            plt.legend(['avg_acc', 'avg_val_acc', 'avg_loss', 'avg_val_loss', 'avg_training_time'], loc='upper left')
            plt.grid(True)
            plt.tight_layout()

            # save png of the plot
            plt.savefig(str(excel_sheet_dir) + "l2_" + str(df_noise['l2_norm_clip'][0]).replace(".", "_"),
                        bbox_inches='tight')
            plt.show()

        if sheet_name not in book.sheetnames:
            book.create_sheet(sheet_name)
        df_res.to_excel(writer, sheet_name=sheet_name, startrow=writer.sheets[sheet_name].max_row, index=True,
                        header=True)
    writer.save()
    return


if __name__ == '__main__':
    main()
