import abc
import os
import numpy as np
import gym
import torch
import random
from copy import deepcopy
from gym import spaces
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device
from matplotlib import animation
import matplotlib.pyplot as plt
import pickle
try:
    from gym.envs.mujoco import MujocoEnv as ContinuousBaseEnv
except:
    from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv as ContinuousBaseEnv

from tuning.utils import Logger

class TuningStrategy(abc.ABC):
    def __init__(self, config, env, device, alg_configs, run_ideal):
        if run_ideal:
            self.result_directory = os.path.join(config['logging_path'], config['env_name'], config['alg_name'], '{}_IDEAL'.format(config['tuning_strategy_name']), str(config['seed']))
        else:
            self.result_directory = os.path.join(config['logging_path'], config['env_name'], config['alg_name'], '{}'.format(config['tuning_strategy_name']), str(config['seed']))

        # Write alg configs
        self.logger = Logger(self.result_directory, run_ideal)
        self.logger.write_initial_files(alg_configs)

        # Store some params on object
        self.num_configs = config['num_configs']
        self.seed = config['seed']
        self.cumulative_exp = 0

        # Store if ideal should be recorded
        self.run_ideal = run_ideal

        # Random initial choice
        self.cfg_index = np.random.choice(self.num_configs)

        # Create seperate eval gym
        self.eval_env = gym.make(config['env_name'])
        self.eval_env.seed(config['seed'])

        # Identify alg class
        if config['alg_name'] == 'PPO':
            self.alg_class = PPO
        elif config['alg_name'] == 'A2C':
            self.alg_class = A2C
        elif config['alg_name'] == 'DQN':
            self.alg_class = DQN
        elif config['alg_name'] == 'SAC':
            self.alg_class = SAC
        else:
            raise Exception('Unsupported algorithm: {}, Supported Algs: [PPO, A2C, DQN, SAC]'.format(config['alg_name']))

        # Determine if off policy or on policy, for interface differences
        self.is_on_policy = issubclass(self.alg_class, OnPolicyAlgorithm)

        # Record on and off policy params
        if self.is_on_policy:
            self.num_steps = config['num_steps']
            
        else:
            self.batch_size = config['batch_size']
            self.gradient_steps = config['num_steps'] // self.batch_size

        # Create models
        self.models = []
        self.callbacks = []
        
        for i, cfg in enumerate(alg_configs):
            if self.is_on_policy:
                model = self.alg_class('MlpPolicy', env, seed=self.seed, verbose=0, n_steps=self.num_steps, device=device, **cfg)

                # Retrieve callback
                buffer_size = self.num_steps * config['num_iters']
                _, callback = model._setup_learn(buffer_size, None)
            else:
                model = self.alg_class('MlpPolicy', env, seed=self.seed, verbose=0, gradient_steps=self.gradient_steps, train_freq=self.batch_size, n_episodes_rollout=-1, device=device, **cfg)

                # Retrieve callback
                _, callback = model._setup_learn(model.buffer_size, None)

            # Store callbacks and models
            self.models.append(model)
            self.callbacks.append(callback)

            # Copy first policy values to all others
            if i == 0:
                pytorch_state_dict = self.models[i].policy.state_dict()
            else:
                self.models[i].policy.load_state_dict(pytorch_state_dict)

    @abc.abstractmethod
    def update(self, batch_index, log=True):
        raise NotImplementedError('Tuning strategy must implement update function.')

    def parent_update(self, batch_index, eval_env):
        if self.run_ideal:
            # Predict ideal selection
            best_index, best_reward = self.calculate_ideal(batch_index, eval_env)

            # Log
            self.logger.log_ideal_selection(batch_index, best_index, best_reward)

    @abc.abstractmethod
    def gather_data(self):
        raise NotImplementedError('Tuning strategy must implement gather_data function.')

    def _run_evaluation(self, model, visualize, eval_env):
        done = False
        obs = eval_env.reset()
        cumulative_reward = 0.0
        frames = []
        while not done:
            if visualize: 
                if issubclass(type(eval_env), ContinuousBaseEnv):
                    frames.append(eval_env.render(mode='rgb_array'))
                else:
                    frames.append(eval_env.render(mode='rgb_array', close=True))

            # Predict
            if self.is_on_policy:
                action, _ = model.predict(obs, deterministic=True)
            else:
                obs = np.reshape(obs, (1, *obs.shape))
                action, _ = model.predict(obs, deterministic=True)
                action = action[0]

            # Interact with env
            next_obs, reward, done, _ = eval_env.step(action, True)

            # Update observation
            obs = next_obs

            # Update reward
            cumulative_reward += reward

        return cumulative_reward, frames

    def evaluate(self, batch_index, visualize, eval_env):
        # Evaluate
        rewards = []
        trajectory_frames = []
        for i in range(10):
            reward, traj_frames = self._run_evaluation(self.models[self.cfg_index], visualize, eval_env)
            rewards.append(reward)
            trajectory_frames.extend(traj_frames)

        # Save trajectories as GIF
        if visualize:
            directory_path = os.path.join(self.result_directory, 'Visualizations')
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            file_path = os.path.join(directory_path, 'eval_traj_{}.gif'.format(batch_index))
            self.save_frames_as_gif(file_path, trajectory_frames)

        # Calculate mean and std dev of reward
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f'Chosen Index:{self.cfg_index}, Mean Reward:{mean_reward:.2f} +/- {std_reward:.2f}')

        # Log selection
        self.logger.log_selection(batch_index, self.cfg_index, self.cumulative_exp, mean_reward, std_reward)

    def calculate_ideal(self, batch_index, eval_env):
        # Save parameters and seed
        np_state = deepcopy(np.random.get_state())
        pytorch_state = deepcopy(torch.random.get_rng_state())
        rand_state = deepcopy(random.getstate())
        eval_env_state = deepcopy(eval_env)

        # Calculate reward for each model
        model_rewards = []
        for i_model in range(len(self.models)):
            # Set parameters and seed for same starting point
            np.random.set_state(deepcopy(np_state))
            torch.random.set_rng_state(deepcopy(pytorch_state))
            random.setstate(deepcopy(rand_state))
            eval_env = eval_env_state
            
            # Evaluate new policy
            rewards = []
            for i in range(10):
                reward, _ = self._run_evaluation(self.models[i_model], False, eval_env)
                rewards.append(reward)
            model_rewards.append(np.mean(rewards))

        # Reset seed
        np.random.set_state(deepcopy(np_state))
        torch.random.set_rng_state(deepcopy(pytorch_state))
        random.setstate(deepcopy(rand_state))
        eval_env = eval_env_state

        # Find arg max
        best_model_index = np.argmax(model_rewards)

        return best_model_index, model_rewards[best_model_index]

    def save_frames_as_gif(self, file_path, frames):
        frame_shape = frames[0].shape
        plt.figure(figsize=(frame_shape[1] / 72.0, frame_shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(file_path, writer='imagemagick', fps=60)

    def save(self, eval_env):
        # Create directory
        directory_path = os.path.join(self.result_directory, 'Checkpoint')
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save eval env from main
        file_path = os.path.join(directory_path, 'eval_env.pkl')
        with open(file_path, 'wb') as o_file:
            pickle.dump(eval_env, o_file, pickle.HIGHEST_PROTOCOL)

        # Save tuning strategy
        file_path = os.path.join(directory_path, 'tuning_strategy.pkl')
        with open(file_path, 'wb') as o_file:
            pickle.dump(self, o_file, pickle.HIGHEST_PROTOCOL)

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

    @staticmethod
    def load(directory_path, device):
        # Load eval env
        file_path = os.path.join(directory_path, 'eval_env.pkl')
        with open(file_path, 'rb') as o_file:
            eval_env = pickle.load(o_file)

        # Load tuning strategy
        file_path = os.path.join(directory_path, 'tuning_strategy.pkl')
        with open(file_path, 'rb') as o_file:
            tuning_strategy = pickle.load(o_file)

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

        for i in range(tuning_strategy.num_configs):
            tuning_strategy.models[i].device = get_device(device)

        return eval_env, tuning_strategy