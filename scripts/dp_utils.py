import dp_accounting
import tensorflow as tf


def get_layers_Binary_Classification():
    return [tf.keras.layers.InputLayer(input_shape=(39,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')]


def get_layers_Linear_Regression():
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
        get_layers_Linear_Regression())

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    model_baseline_linear.compile(optimizer=optimizer,
                                  loss='mean_squared_error',
                                  metrics='accuracy')

    model.append(model_baseline_binary)
    model.append(model_baseline_linear)
    return model


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