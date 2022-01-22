import os
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from tuning.utils import create_configs
from .base import TuningStrategy

class RandomFlex(TuningStrategy):
    def __init__(self, config, env, device):
        # Create configs and write to file
        configs = create_configs(config['alg_name'], config['num_configs'], config['n_epochs'])
        super().__init__(config, env, device, configs, False)

    def update(self, batch_index, log=True):
        # Cache original weights
        orig_weights = self.models[self.cfg_index].policy.state_dict()

        # Train models
        for j in range(self.num_configs):
            # Set rollout for off policy learners
            if j != self.cfg_index:
                if self.is_on_policy:
                    self.models[j]._last_obs = self.models[self.cfg_index]._last_obs
                    self.models[j].rollout_buffer = self.models[self.cfg_index].rollout_buffer
                else:
                    self.models[j]._last_obs = self.models[self.cfg_index]._last_obs
                    self.models[j].replay_buffer = self.models[self.cfg_index].replay_buffer
                
                # Copy current policy to all others
                self.models[j].policy.load_state_dict(orig_weights)

        # Random choice for next HP config
        selected_index = np.random.randint(0, self.num_configs)
        if log:
            self.cfg_index = selected_index

        return selected_index

    def gather_data(self):
        # Collect rollouts
        if self.is_on_policy:
            self.models[self.cfg_index].collect_rollouts(
                self.models[self.cfg_index].env, 
                self.callbacks[self.cfg_index], 
                self.models[self.cfg_index].rollout_buffer, 
                self.num_steps
            )

            # Record experience
            self.cumulative_exp += self.num_steps
        else:
            self.models[self.cfg_index].collect_rollouts(
                self.models[self.cfg_index].env,
                n_episodes=self.models[self.cfg_index].n_episodes_rollout,
                n_steps=self.models[self.cfg_index].train_freq,
                action_noise=self.models[self.cfg_index].action_noise,
                callback=self.callbacks[self.cfg_index],
                learning_starts=self.models[self.cfg_index].learning_starts,
                replay_buffer=self.models[self.cfg_index].replay_buffer,
                log_interval=None
            )

            # Record experience
            self.cumulative_exp += self.models[self.cfg_index].train_freq

