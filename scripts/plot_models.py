from keras.src.utils import plot_model
from tensorflow_privacy import DPSequential
from tensorflow_privacy.privacy.keras_models import dp_keras_model
import tensorflow as tf

from scripts.dp_ml_utils import get_layers_logistic_regression, get_layers_dnn


def main():
    dp_sgd_model = dp_keras_model.DPSequential(
        l2_norm_clip=6,
        noise_multiplier=11,
        num_microbatches=1,  # wandb.config.microbatches
        layers=get_layers_dnn())
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00009455)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    dp_sgd_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    plot_model(dp_sgd_model,

               to_file='../Plots/Models/dpsgd_model.png',

               show_shapes=True,

               show_layer_names=True)

    dp_logreg_model = DPSequential(
        l2_norm_clip=6,
        noise_multiplier=11,
        num_microbatches=1,
        layers=get_layers_logistic_regression())

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.000001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    dp_logreg_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    plot_model(dp_logreg_model,

               to_file='../Plots/Models/dp_logreg_model.png',

               show_shapes=True,

               show_layer_names=True)


if __name__ == '__main__':
    main()
