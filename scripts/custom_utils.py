import os

import dp_accounting
import pandas as pd
import matplotlib.pyplot as plt
import wandb

""" Computes epsilon value for given hyperparameters """
def compute_epsilon_noise(steps, noise_multiplier, batch_size, num_samples):
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    accountant = dp_accounting.rdp.RdpAccountant(orders)

    sampling_probability = batch_size / num_samples
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(noise_multiplier)), steps)

    accountant.compose(event)

    # Delta is set to 1e-6 because dataset has 180,000 training points.
    return accountant.get_epsilon(target_delta=1e-6)


def log_metrics_dp(history, m, index, actual_dir, model, optimizer):
    history_dict = history.history
    history_df = pd.DataFrame(history_dict)
    setup_dir = model + '_' + optimizer + '/'
    if not os.path.exists(actual_dir + setup_dir):
        os.makedirs(actual_dir + setup_dir)

    # summarize history for metrics
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
        'DP Model (' + optimizer + ') acc and loss with noise: ' + str(
            m['noise_multiplier']) + ', l2_norm_clip: ' + str(m['l2_norm_clip']))
    plt.ylabel('loss, acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # save png of the plot

    plt.savefig(str(actual_dir) + "dp_n_" + str(m['noise_multiplier']).replace(".", "_")
                + "l2_" + str(m['l2_norm_clip']).replace(".", "_"), bbox_inches='tight')
    plt.show()

    title = "Accuracy and Loss metrics of DP model (" + optimizer \
            + ") training: noise=" + (str(m['noise_multiplier']) + ", l2_norm_clip=" + str(m['l2_norm_clip']))

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


def log_metrics_baseline(history, index, actual_dir, batch_size, microbatches):
    history_dict = history.history
    history_df = pd.DataFrame(history_dict)
    baseline_dir = 'baseline/'
    if not os.path.exists(actual_dir + baseline_dir):
        os.makedirs(actual_dir + baseline_dir)

    # summarize history for accuracy
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

    plt.ylabel('loss, acc')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if index == 0:
        plt.title('Binary model Accuracy and Loss metrics with batch_size = ' + str(batch_size) +
                  ', microbatches = ' + str(microbatches) + ", ")

        plt.savefig(str(actual_dir) + "baseline_binary_acc", bbox_inches='tight')
        wandb.log(
            {"Accuracy and Loss metrics of baseline Binary Classification model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["accuracy"],
                        history_df["val_accuracy"]],
                    keys=["accuracy", "val_accuracy"],
                    title="Accuracy metrics of baseline Binary Classification model",
                    xname="epochs"
                )})

    else:
        plt.title('Linear model Accuracy and Loss metrics with batch_size = ' + str(batch_size) +
                  ', microbatches = ' + str(microbatches) + ", ")

        plt.savefig(str(actual_dir) + "baseline_linear_acc", bbox_inches='tight')
        wandb.log(
            {"Accuracy and Loss metrics of baseline Binary Classification model":
                wandb.plot.line_series(
                    xs=history_df.index,
                    ys=[history_df["accuracy"],
                        history_df["val_accuracy"]],
                    keys=["accuracy", "val_accuracy"],
                    title="Accuracy metrics of baseline Linear regression model",
                    xname="epochs")})
    plt.show()
