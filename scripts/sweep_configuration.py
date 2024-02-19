metric = {
    "name": "loss",
    "goal": "minimize"
}

parameters = {
    "learning_rate": {"values": [0.00001]},
    "batch_size": {"values": [30]},
    "epochs": {"values": [6]},
    "l2_norm_clip": {"values": [0.15, 0.30, 0.6, 0.9, 1.2, 1.5, 2]},
    "noise_multiplier": {"values": [0.1, 0.30, 0.6, 0.9, 1.2, 1.5, 2]}
}

parameters_random = {
    "learning_rate": {"values": [0.00001]},
    "batch_size": {"values": [30]},
    "epochs": {"values": [6]},
    "l2_norm_clip": {"values": [0.15, 0.30, 0.45, 0.6, 0.75, 0.9, 1.2, 1.5, 1.75, 2, 2.5, 3, 4, 7, 10]},
    "noise_multiplier": {"values": [0.1, 0.30, 0.45, 0.6, 0.75, 0.9, 1.2, 1.5, 2, 2.5, 3, 5]}
}

sweep_configuration_bayes = {
    "method": "bayes",  # random, grid
    "metric": metric,
    "parameters": parameters
}

sweep_configuration_random = {
    "method": "random",
    "parameters": parameters_random
}
