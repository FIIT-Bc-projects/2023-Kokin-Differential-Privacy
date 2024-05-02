metric_loss = {
    'name': 'loss',
    'goal': 'minimize'
}

metric_acc = {
    'name': 'accuracy',
    'goal': 'maximize'
}

metric_val_accuracy = {
    'name': 'val_accuracy',
    'goal': 'maximize'
}

metric_eps = {
    'name': 'epsilon',
    'goal': 'minimize'
}

# https://wandb.ai/dpnerds/sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning/runs/2c1t7i2w?workspace=user-xkokin
parameters_grid_dp_sgd = {
    'learning_rate': {'values': [0.00009455]},
    'batch_size': {'values': [30]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [0.15, 0.6, 1.2, 2, 5, 10, 20]},
    'noise_multiplier': {'values': [0.1, 0.50, 0.9, 1.5, 5, 10, 20]},
    'microbatches': {'values': [1, 15, 30]}
}

parameters_grid_dp_dnn_patterns = {
    'learning_rate': {'values': [0.0001]},
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [0.1, 1, 5, 10, 20]},
    'noise_multiplier': {'values': [0.01, 0.1, 1, 5, 10, 100]},
    'microbatches': {'values': [1]}
}

parameters_grid_dp_logreg_patterns = {
    'learning_rate': {'values': [0.0001]},
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [0.1, 1, 5, 10, 20]},
    'noise_multiplier': {'values': [0.01, 0.1, 1, 5, 10, 100]},
    'microbatches': {'values': [1]}
}

parameters_bayes_random_dp_dnn = {
    'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [0.1, 1, 5, 10]},
    'noise_multiplier': {'values': [0.01, 0.1, 0.5, 1, 5, 10]},
    'microbatches': {'values': [1, 15]}
}

parameters_bayes_dp_learning_rate = {
    'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [5, 10]},
    'noise_multiplier': {'values': [0.35, 0.5, 1, 5, 10]},
    'microbatches': {'values': [1, 15]}
}

parameters_grid_dp_microbatches = {
    'learning_rate': {'values': [0.0001]},
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'l2_norm_clip': {'values': [5]},
    'noise_multiplier': {'values': [0.35, 0.5, 1, 5, 10, 20, 50, 100]},
    'microbatches': {'values': [1, 3, 5, 15]}
}

parameters_grid_baseline = {
    'learning_rate': {'values': [0.001, 0.0001, 0.00005, 0.00001, 0.000001]},
    'batch_size': {'values': [15, 30, 60, 90, 180]},
    'epochs': {'values': [3, 5, 7, 9, 12]},
}

parameters_bayes_random_baseline = {
    'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]},
    'batch_size': {'values': [15, 30, 60, 90]},
    'epochs': {'values': [3, 5, 7, 9, 12, 15, 20]}

}

parameters_grid_epsilon = {
    'batch_size': {'values': [15, 30]},
    'epochs': {'values': [5]},
    'noise_multiplier': {'values': [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 3, 5, 10, 20]},
}

parameters_bayes_logreg_dp = {
    # On the base of the best results from the baseline
    # https://wandb.ai/dpnerds/sweeps-hyperparameter-tuning-in-privacy-preserving-machine-learning/runs/vif1evos?workspace=user-xkokin
    'batch_size': {'values': [15]},
    'epochs': {'values': [5]},
    'learning_rate': {'values': [0.01, 0.001, 0.0001, 0.00001]},
    'l2_norm_clip': {'values': [0.1, 1, 5, 10]},
    'noise_multiplier': {'values': [0.01, 0.1, 0.5, 1, 5, 10]},
    'microbatches': {'values': [1, 15]}
}

sweep_configuration_grid_dp_dnn = {
    'name': 'dp_dnn_grid_search',
    'method': 'grid',  # random, grid
    'metric': metric_acc,
    'parameters': parameters_grid_dp_sgd
}

sweep_configuration_bayes_logreg_dp = {
    'name': 'dp_logreg_bayes_search_hyperband',
    'method': 'bayes',  # random, grid
    'metric': metric_eps,
    'parameters': parameters_bayes_logreg_dp,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_bayes_dnn_dp = {
    'name': 'dp_sgd_bayes_search_hyperband',
    'method': 'bayes',  # random, grid
    'metric': metric_eps,
    'parameters': parameters_bayes_random_dp_dnn,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_bayes_logreg_dp_learning_rate = {
    'name': 'dp_logreg_bayes_search_hyperband_learing_rate',
    'method': 'bayes',  # random, grid
    'metric': metric_val_accuracy,
    'parameters': parameters_bayes_dp_learning_rate,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_bayes_dnn_dp_learning_rate = {
    'name': 'dp_dnn_bayes_search_hyperband_learning_rate',
    'method': 'bayes',  # random, grid
    'metric': metric_val_accuracy,
    'parameters': parameters_bayes_dp_learning_rate,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_grid_logreg_dp_microbatches = {
    'name': 'dp_logreg_grid_search_hyperband_microbatches',
    'method': 'grid',
    'metric': metric_val_accuracy,
    'parameters': parameters_grid_dp_microbatches,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_grid_dnn_dp_microbatches = {
    'name': 'dp_dnn_grid_search_hyperband_microbatches',
    'method': 'grid',
    'metric': metric_val_accuracy,
    'parameters': parameters_grid_dp_microbatches,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_random_dp_sgd_dnn = {
    'name': 'dp_sgd_random_search',
    'method': 'random',
    'metric': metric_acc,
    'parameters': parameters_bayes_random_dp_dnn
}

sweep_configuration_bayes_baseline_dnn = {
    'name': 'baseline_sgd_bayes_search_hyperband',
    'method': 'bayes',
    'metric': metric_acc,
    'parameters': parameters_bayes_random_baseline,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_grid_baseline_dnn = {
    'name': 'baseline_dnn_grid_search',
    'method': 'grid',
    'metric': metric_acc,
    'parameters': parameters_grid_baseline
}

sweep_configuration_bayes_baseline_logreg = {
    'name': 'baseline_logreg_bayes_search_hyperband',
    'method': 'bayes',
    'metric': metric_acc,
    'parameters': parameters_bayes_random_baseline,
    'early_terminate': {'type': 'hyperband',
                        'min_iter': 3
                        }
}

sweep_configuration_grid_baseline_logreg = {
    'name': 'baseline_logreg_grid_search',
    'method': 'grid',
    'metric': metric_acc,
    'parameters': parameters_grid_baseline
}

sweep_configuration_epsilon_grid = {
    'name': 'epsilon_grid_search',
    'method': 'grid',
    'metric': metric_eps,
    'parameters': parameters_grid_epsilon
}

sweep_configuration_grid_dp_dnn_patterns = {
    'name': 'dp_dnn_patterns_search',
    'method': 'grid',
    'metric': metric_acc,
    'parameters': parameters_grid_dp_dnn_patterns
}

sweep_configuration_grid_dp_logreg_patterns = {
    'name': 'dp_logreg_patterns_search',
    'method': 'grid',
    'metric': metric_acc,
    'parameters': parameters_grid_dp_logreg_patterns
}
