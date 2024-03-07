import time

import pandas as pd
# import wandb
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.keras_models import dp_keras_model
import tensorflow as tf



from scripts.dataset_utils import get_preprocessed_data
from scripts.dp_utils import get_layers_dnn, compute_epsilon_noise


# class PrivacyMetrics(tf.keras.callbacks.Callback):
#   def __init__(self, epochs_per_report, model_name):
#     self.epochs_per_report = epochs_per_report
#     self.model_name = model_name
#     self.attack_results = []
#
#   def on_epoch_end(self, epoch, logs=None):
#     epoch = epoch+1
#
#     if epoch % self.epochs_per_report != 0:
#       return
#
#     print(f'\nRunning privacy report for epoch: {epoch}\n')
#
#     logits_train = self.model.predict(x_train, batch_size=batch_size)
#     logits_test = self.model.predict(x_test, batch_size=batch_size)
#
#     prob_train = special.softmax(logits_train, axis=1)
#     prob_test = special.softmax(logits_test, axis=1)
#
#     # Add metadata to generate a privacy report.
#     privacy_report_metadata = PrivacyReportMetadata(
#         # Show the validation accuracy on the plot
#         # It's what you send to train_accuracy that gets plotted.
#         accuracy_train=logs['val_accuracy'],
#         accuracy_test=logs['val_accuracy'],
#         epoch_num=epoch,
#         model_variant_label=self.model_name)
#
#     attack_results = mia.run_attacks(
#         AttackInputData(
#             labels_train=y_train_indices[:, 0],
#             labels_test=y_test_indices[:, 0],
#             probs_train=prob_train,
#             probs_test=prob_test),
#         SlicingSpec(entire_dataset=True, by_class=True),
#         attack_types=(AttackType.THRESHOLD_ATTACK,
#                       AttackType.LOGISTIC_REGRESSION),
#         privacy_report_metadata=privacy_report_metadata)
#
#     self.attack_results.append(attack_results)

def main():
    noise_mul = 2
    epochs = 5
    batch_size = 30
    microbatches = 1
    l2_norm_clip = 7
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning"
    # )
    """ Prepare and split data"""
    x, y = get_preprocessed_data()

    # Take a look at the number of rows
    # print("Data shape: " + str(x.shape))
    # print("Labels shape: " + str(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2682926)
    # print("X_train shape: " + str(x_train.shape))
    # print("X_test shape: " + str(x_test.shape))

    total_samples = x_train.shape[0]
    print("Total samples: ", total_samples)

    model = dp_keras_model.DPSequential(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_mul,
        num_microbatches=microbatches,  # wandb.config.microbatches
        layers=get_layers_dnn())
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size)
    end_time = time.time()

    epsilon = compute_epsilon_noise(5 * x_train.shape[0] // batch_size,
                                    noise_mul,
                                    batch_size, total_samples)

    # wandb.log({"Epsilon": epsilon})
    print("Epsilon: " + str(epsilon))

    training_time = round(end_time - start_time, 4)
    # wandb.log({"Training_time": training_time})
    print("Training time: " + str(training_time))

    hist_df = pd.DataFrame(history.history)

    for index, epoch in hist_df.iterrows():
        print({'epochs': index + 1,
               'loss': round(hist_df['loss'][index], 4),
               'acc': round(hist_df['accuracy'][index], 4),
               'val_loss': round(hist_df['val_loss'][index], 4),
               'val_acc': round(hist_df['val_accuracy'][index], 4)
               })
        # wandb.log({'epochs': index,
        #            'loss': round(hist_df['loss'][index], 4),
        #            'accuracy': round(hist_df['accuracy'][index], 4),
        #            'val_loss': round(hist_df['val_loss'][index], 4),
        #            'val_accuracy': round(hist_df['val_accuracy'][index], 4)
        #            })


if __name__ == '__main__':
    main()
