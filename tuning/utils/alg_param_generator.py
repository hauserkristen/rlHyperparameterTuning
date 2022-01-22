import numpy as np
from typing import List
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling

# All
gamma_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
learning_rate_range = [1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3]

# A2C, PPO, SAC
ent_coeff_range = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]

# A2C and PPO
vf_coeff_range = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]
gae_lambda_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# PPO
clip_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
clip_vf_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

# DQN
epsilon = [0.05, 0.1, 0.15, 0.2]
epsilon_decay = [0.9, 0.95, 0.99, 0.999]

# SAC
tau_range = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075]

def create_configs(alg_name, num_configs, n_epochs):
    # Create options
    possible_alg_param_sets = __get_alg_param_set_indicies(alg_name)

    # Select subset via latin hypercube
    lhs = LatinHypercubeSampling(possible_alg_param_sets, num_configs, sampling_type='selection') 

    # Retrieve configs
    alg_param_samples = lhs.sample_points()

    # Construct configs
    return __construct_alg_configs(alg_name, alg_param_samples, n_epochs)

def __get_alg_param_set_indicies(alg_name):
    possible_param_sets = []
    if alg_name == 'DQN':
        for df in gamma_range:
            for lr in learning_rate_range:
                # Constant epsilon
                for e in epsilon:
                    possible_param_sets.append([df, lr, 1.0, e, e])

                # Degarding epsilon
                for e in epsilon_decay:
                    possible_param_sets.append([df, lr, e, 1.0, 0.05])
    elif alg_name == 'A2C':
        for df in gamma_range:
            for lr in learning_rate_range:
                for ec in ent_coeff_range:
                    for vc in vf_coeff_range:
                        for gae in gae_lambda_range:
                            possible_param_sets.append([df, lr, ec, vc, gae])
    elif alg_name == 'PPO':
        for df in gamma_range:
            for lr in learning_rate_range:
                for ec in ent_coeff_range:
                    for vc in vf_coeff_range:
                        for gae in gae_lambda_range:
                            for cl in clip_range:
                                for cl_vf in clip_vf_range:
                                    possible_param_sets.append([df, lr, ec, vc, gae, cl, cl_vf])
    elif alg_name == 'SAC':
        for df in gamma_range:
            for lr in learning_rate_range:
                for ec in ent_coeff_range:
                    for t in tau_range:
                        possible_param_sets.append([df, lr, ec, t])
    else:
        raise Exception('Unsupported RL aglorithm: {}'.format(alg_name))

    return np.array(possible_param_sets)

def __construct_alg_configs(alg_name, alg_param_samples, n_epochs):
    configs = []
    if alg_name == 'DQN':
        for df, lr, eps, eps_max, eps_min in alg_param_samples:
            config = {
                'gamma': df,
                'learning_rate': lr,
                'exploration_fraction': eps,
                'exploration_initial_eps': eps_max,
                'exploration_final_eps': eps_min
            }

            configs.append(config)
    elif alg_name == 'A2C':
        for df, lr, ec, vc, gae in alg_param_samples:
            config = {
                'gamma': df,
                'learning_rate': lr,
                'gae_lambda': gae,
                'ent_coef': ec,
                'vf_coef': vc
            }

            configs.append(config)
    elif alg_name == 'PPO':
        for df, lr, ec, vc, gae, cl, cl_vf in alg_param_samples:
            config = {
                'gamma': df,
                'learning_rate': lr,
                'gae_lambda': gae,
                'ent_coef': ec,
                'vf_coef': vc,
                'clip_range': cl,
                'clip_range_vf': cl_vf,
                'n_epochs': n_epochs
            }

            configs.append(config)
    elif alg_name == 'SAC':
        for df, lr, ec, tau in alg_param_samples:
            config = {
                'gamma': df,
                'learning_rate': lr,
                'ent_coef': ec,
                'tau': tau
            }

            configs.append(config)
    else:
        raise Exception('Unsupported RL aglorithm: {}'.format(alg_name))

    return configs