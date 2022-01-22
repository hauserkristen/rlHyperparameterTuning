import os
import gym
import random
import json
import argparse
from parse import search
import numpy as np
import pybulletgym
import torch

from tuning import create_tuning_strategy, load_tuning_strategy
from utils import *

# String constants
IDEAL_FRMT = '{tuning_strat}-IDEAL'

def create_exp_config(args):
    # Read exp config
    cfg_path = os.path.join('configs', 'experiment_params.json')
    with open(cfg_path) as f:
        exp_config = json.load(f)

    # Construct subconfig
    config = {
        'seed': args.seed,
        'tuning_strategy_name': args.tuning_strategy_name,
        'alg_name': args.alg_name,
        'env_name': args.env_name,
        'num_iters': exp_config['num_iters'],
        'logging_path': exp_config['logging_path']
    }

    # Add environemtn params
    for k, v in exp_config['env_params'][args.env_name].items():
        config[k] = v

    # Add tuning strategy params
    if args.tuning_strategy_name == 'SEHOP' or args.tuning_strategy_name == 'PBT':
        for k, v in exp_config['tuning_strategy_params'][args.tuning_strategy_name][args.env_name].items():
            config[k] = v
    else:
        for k, v in exp_config['tuning_strategy_params'][args.tuning_strategy_name].items():
            config[k] = v
    
    
    return config

def main():
    # Create command line arguments
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning on an RL experiment.')
    parser.add_argument('seed', help='Random seed for the experiment', type=int)
    parser.add_argument('tuning_strategy_name', help='Tuning strategy to use, [SEHOP, HOOF, PBT].')
    parser.add_argument('alg_name', help='Algorithm to tune, [PPO, A2C, DQN, SAC].')
    parser.add_argument('env_name', help='Environment to tune on, [CustomLunarLander-v0, CustomCartPole-v0, CustomReacher-v0].')
    parser.add_argument('-v', '--visualize', help='Visualize and save evaluation trajectories.', action="store_true")
    parser.add_argument('-s', '--save', help='Save model.', action="store_true")
    parser.add_argument('-l', '--load', help='Loads previous save point.', action="store_true")
    args = parser.parse_args()

    # Prase run ideal from alg name
    ideal_search = search(IDEAL_FRMT, args.tuning_strategy_name)
    run_ideal = ideal_search is not None
    if run_ideal:
        args.tuning_strategy_name = ideal_search['tuning_strat']

    # Create config
    config = create_exp_config(args)

    # Identify device to be used
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(device)

    if args.load:
        if run_ideal:
            directory_path = os.path.join(config['logging_path'], config['env_name'], config['alg_name'], '{}_IDEAL'.format(config['tuning_strategy_name']), str(config['seed']))
        else:
            directory_path = os.path.join(config['logging_path'], config['env_name'], config['alg_name'], '{}'.format(config['tuning_strategy_name']), str(config['seed']))

        eval_env, tuning_strategy = load_tuning_strategy(config, directory_path, device)
    else:
        # Create envs
        env = gym.make('ht_envs:{}'.format(config['env_name']))
        eval_env = gym.make('ht_envs:{}'.format(config['env_name']))

        # Set seeds
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        env.seed(config['seed'])
        eval_env.seed(config['seed'])

        # Create tuning strategy
        tuning_strategy = create_tuning_strategy(config, env, device, run_ideal)

    # Visualization and save interval
    VISUALIZATION_INTERVAL = 500
    SAVE_INTERVAL = 100

    # Iterate
    for i in range(config['num_iters']):
        tuning_strategy.gather_data()

        tuning_strategy.update(i)
        tuning_strategy.parent_update(i, eval_env)

        visualize = i % VISUALIZATION_INTERVAL == 0 and args.visualize
        tuning_strategy.evaluate(i, visualize, eval_env)

        if args.save and i % SAVE_INTERVAL == 0:
            tuning_strategy.save(eval_env)

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()