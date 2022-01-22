import os
import gym
import random
import json
import argparse
from parse import search
import numpy as np
import pybulletgym
from torch import manual_seed
from gym import spaces
import pickle
import torch
from shutil import copyfile

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from tuning.utils import Logger

def create_exp_config(args):
    # Read exp config
    cfg_path = os.path.join('configs', 'hand_tuned_params.json')
    with open(cfg_path) as f:
        exp_config = json.load(f)

    # Construct subconfig
    config = {
        'seed': args.seed,
        'alg_name': args.alg_name,
        'env_name': args.env_name,
        'num_iters': exp_config['num_iters'],
        'logging_path': exp_config['logging_path']
    }

    # Add environment params
    alg_config= {}
    for k, v in exp_config['env_params'][args.env_name].items():
        alg_config[k] = v

    # Add algorithm params
    for k, v in exp_config['alg_params'][args.alg_name][args.env_name].items():
        alg_config[k] = v

    return config, alg_config

def run_evaluation(model, eval_env, is_on_policy):
        done = False
        obs = eval_env.reset()
        cumulative_reward = 0.0
        while not done:
            # Predict
            if is_on_policy:
                action, _ = model.predict(obs, deterministic=True)
            else:
                obs = np.reshape(obs, (1, *obs.shape))
                action, _ = model.predict(obs, deterministic=True)
                action = action[0]

            # Format action
            if isinstance(model.action_space, spaces.Discrete):
                action_log = np.array([action])
            else:
                action_log = action

            # Interact with env
            next_obs, reward, done, _ = eval_env.step(action, True)

            # Update observation
            obs = next_obs

            # Update reward
            cumulative_reward += reward

        return cumulative_reward

def evaluate(model, eval_env, is_on_policy):
        # Evaluate
        rewards = [run_evaluation(model, eval_env, is_on_policy) for i in range(10)]

        # Calculate mean and std dev of reward
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"Mean Reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

def update_hp_config(alg_config, alg_index):
    # Read file
    file_path = os.path.join('results', 'CustomLunarLander-v0', 'DQN', 'RANDOM_FIXED', '2', 'alg_def.csv')
    hp_config_values = {}
    header = []
    with open(file_path) as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            row_split = [x.strip() for x in row_split]
               
            if row_index == 0:               
                # Read header
                header = row_split
            elif row_index == int(alg_index):
                # Read values
                for key, value in zip(header, row_split):
                    hp_config_values[key] = value
                
            row_index += 1

    # Replace values
    for key, value in hp_config_values.items():
        if key in alg_config.keys():
            alg_config[key] = float(value)

    return alg_config

def main():
    # Create command line arguments
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning on an RL experiment.')
    parser.add_argument('seed', help='Random seed for the experiment', type=int)
    parser.add_argument('alg_name', help='Algorithm to tune, [PPO, A2C, DQN, SAC].')
    parser.add_argument('env_name', help='Environment to tune on, [CustomLunarLander-v0, CustomCartPole-v0, CustomReacher-v0].')
    parser.add_argument('alg_index', help='Support to help DQN-LunarLander random baseline.')
    parser.add_argument('-l', '--load', help='Loads previous save point.', action="store_true")
    args = parser.parse_args()

    # Create config
    config, alg_config = create_exp_config(args)

    # Create envs
    env = gym.make('ht_envs:{}'.format(config['env_name']))
    eval_env = gym.make('ht_envs:{}'.format(config['env_name']))

    # Create directory
    result_directory = os.path.join(config['logging_path'], config['env_name'], config['alg_name'], 'HAND_TUNED_{}'.format(args.alg_index), str(config['seed']))

    # Identify alg class
    if config['alg_name'] == 'PPO':
        alg_class = PPO
    elif config['alg_name'] == 'A2C':
        alg_class = A2C
    elif config['alg_name'] == 'DQN':
        alg_class = DQN
    elif config['alg_name'] == 'SAC':
        alg_class = SAC
    else:
        raise Exception('Unsupported algorithm: {}, Supported Algs: [PPO, A2C, DQN, SAC]'.format(config['alg_name']))

    # Determine if off policy or on policy, for interface differences
    is_on_policy = issubclass(alg_class, OnPolicyAlgorithm)

    if args.load:
        directory_path = os.path.join(result_directory, 'Checkpoint')

        # Create logger reference
        logger = Logger(result_directory, False, False)

        # Copy selection file
        checkpoint_file = os.path.join(directory_path, 'alg_sel.csv')
        current_file = os.path.join(result_directory, 'alg_sel.csv')
        copyfile(checkpoint_file, current_file)

        # Load model
        file_path = os.path.join(directory_path, 'model')
        model = alg_class.load(file_path)

        # Create callback
        if is_on_policy:
            num_steps = alg_config['num_steps']
            buffer_size = num_steps * config['num_iters']
            callback = model._init_callback(None, None, 10000, 5, None)
        else:
            batch_size = alg_config['batch_size']
            gradient_steps = alg_config['num_steps'] // batch_size
            callback = model._init_callback(None, None, 10000, 5, None)

        # Load iteration values
        file_path = os.path.join(directory_path, 'iter_vals.pkl')
        with open(file_path, 'rb') as o_file:
            iter_vals = pickle.load(o_file)
        i = iter_vals['i'] + 1
        cumulative_exp = iter_vals['cumulative_exp']
        
        # Load env
        file_path = os.path.join(directory_path, 'env.pkl')
        with open(file_path, 'rb') as o_file:
            env = pickle.load(o_file)
        model.set_env(env)

        # Load eval env
        file_path = os.path.join(directory_path, 'eval_env.pkl')
        with open(file_path, 'rb') as o_file:
            eval_env = pickle.load(o_file)

        # Load seeds
        file_path = os.path.join(directory_path, 'numpy_seed.pkl')
        with open(file_path, 'rb') as o_file:
            seed_state = pickle.load(o_file)
            np.random.set_state(seed_state)

        file_path = os.path.join(directory_path, 'random_seed.pkl')
        with open(file_path, 'rb') as o_file:
            seed_state = pickle.load(o_file)
            random.setstate(seed_state)

        file_path = os.path.join(directory_path, 'torch_seed.pkl')
        with open(file_path, 'rb') as o_file:
            seed_state = pickle.load(o_file)
            torch.set_rng_state(seed_state)
    else:
        # Set start iteration counter and experience
        i = 0
        cumulative_exp = 0

        # Read lunarlander alg def file and update values
        alg_config = update_hp_config(alg_config, args.alg_index)

        # Set seeds
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        manual_seed(config['seed'])
        env.seed(config['seed'])
        eval_env.seed(config['seed'])

        # Write alg configs
        logger = Logger(result_directory, False)
        logger.write_initial_files([config])

        # Record on and off policy params
        if is_on_policy:
            num_steps = alg_config['num_steps']
            del alg_config['num_steps']
            del alg_config['batch_size']
            if alg_class is not PPO:
                del alg_config['n_epochs']
        else:
            batch_size = alg_config['batch_size']
            gradient_steps = alg_config['num_steps'] // batch_size
            del alg_config['num_steps']
            del alg_config['batch_size']
            del alg_config['n_epochs']

        if is_on_policy:
            model = alg_class('MlpPolicy', env, seed=config['seed'], verbose=0, n_steps=num_steps, **alg_config)

            # Retrieve callback
            buffer_size = num_steps * config['num_iters']
            _, callback = model._setup_learn(buffer_size, None)
        else:
            model = alg_class('MlpPolicy', env, seed=config['seed'], verbose=0, gradient_steps=gradient_steps, train_freq=batch_size, n_episodes_rollout=-1, **alg_config)

            # Retrieve callback
            _, callback = model._setup_learn(model.buffer_size, None)

    # Save interval
    SAVE_INTERVAL = 10

    # Iterate
    while i < config['num_iters']:
        # Collect rollouts
        if is_on_policy:
            model.collect_rollouts(
                model.env, 
                callback, 
                model.rollout_buffer, 
                num_steps
            )

            # Record experience
            cumulative_exp += num_steps
        else:
            model.collect_rollouts(
                model.env,
                n_episodes=model.n_episodes_rollout,
                n_steps=model.train_freq,
                action_noise=model.action_noise,
                callback=callback,
                learning_starts=model.learning_starts,
                replay_buffer=model.replay_buffer,
                log_interval=None
            )

            # Record experience
            cumulative_exp += model.train_freq

        # Train model
        if is_on_policy:
            model.train()
        else:
            model.train(gradient_steps=gradient_steps, batch_size=batch_size)

        # Evaluate
        mean_reward, std_reward = evaluate(model, eval_env, is_on_policy)

        # Log results
        logger.log_selection(i, 0, cumulative_exp, mean_reward, std_reward)

        if i % SAVE_INTERVAL == 0:
            # Create directory
            directory_path = os.path.join(result_directory, 'Checkpoint')
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Save model
            file_path = os.path.join(directory_path, 'model')
            model.save(file_path)

            # Save env
            file_path = os.path.join(directory_path, 'env.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump(env, o_file, pickle.HIGHEST_PROTOCOL)

            # Save eval env
            file_path = os.path.join(directory_path, 'eval_env.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump(eval_env, o_file, pickle.HIGHEST_PROTOCOL)

            # Save seeds
            file_path = os.path.join(directory_path, 'numpy_seed.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump(np.random.get_state(), o_file, pickle.HIGHEST_PROTOCOL)

            file_path = os.path.join(directory_path, 'random_seed.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump(random.getstate(), o_file, pickle.HIGHEST_PROTOCOL)

            file_path = os.path.join(directory_path, 'torch_seed.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump(torch.get_rng_state(), o_file, pickle.HIGHEST_PROTOCOL)

            # Save iteration values
            file_path = os.path.join(directory_path, 'iter_vals.pkl')
            with open(file_path, 'wb') as o_file:
                pickle.dump({'i': i, 'cumulative_exp': cumulative_exp}, o_file, pickle.HIGHEST_PROTOCOL)

            # Copy selection file
            current_file = os.path.join(result_directory, 'alg_sel.csv')
            if os.path.exists(current_file):
                destination_file = os.path.join(directory_path, 'alg_sel.csv')
                copyfile(current_file, destination_file)

        # Increment iteration
        i += 1

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()