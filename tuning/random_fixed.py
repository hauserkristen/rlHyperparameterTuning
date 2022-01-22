import os
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from tuning.utils import create_configs
from .base import TuningStrategy

class RandomFixed(TuningStrategy):
    def __init__(self, config, env, device):
        # Create configs and write to file
        configs = create_configs(config['alg_name'], config['num_configs'], config['n_epochs'])
        super().__init__(config, env, device, configs, False)

        # Create filname and remove if previously existed
        self.result_file = os.path.join(self.result_directory, 'alg_result.csv')
        if os.path.exists(self.result_file):
            os.remove(self.result_file)

        # Write header
        column_header = ['Reward {}'.format(x) for x in range(config['num_configs'])]
        column_header.insert(0, 'Episode Number')
        with open(self.result_file, mode='w') as f:
            # Write header
            header = ','.join(column_header)
            f.write(header + '\n')

    def update(self, batch_index, log=True):
        # Train models
        for j in range(self.num_configs):
            # Train model
            if self.is_on_policy:
                self.models[j].train()
            else:
                self.models[j].train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

    def gather_data(self):
        # Collect rollouts for all configurations
        for i in range(self.num_configs):
            if self.is_on_policy:
                self.models[i].collect_rollouts(
                    self.models[i].env, 
                    self.callbacks[i], 
                    self.models[i].rollout_buffer, 
                    self.num_steps
                )
      
            else:
                self.models[i].collect_rollouts(
                    self.models[i].env,
                    n_episodes=self.models[i].n_episodes_rollout,
                    n_steps=self.models[i].train_freq,
                    action_noise=self.models[i].action_noise,
                    callback=self.callbacks[i],
                    learning_starts=self.models[i].learning_starts,
                    replay_buffer=self.models[i].replay_buffer,
                    log_interval=None
                )
        # Record experience
        if self.is_on_policy:
            self.cumulative_exp += self.num_steps
        else:
            self.cumulative_exp += self.models[i].train_freq

    def evaluate(self, batch_index, visualize, eval_env):
        # Evaluate every configuration
        rewards = []
        for j_config in range(self.num_configs):
            eval_results = [self._run_evaluation(self.models[j_config], False, eval_env) for i in range(10)]
            r = [reward for (reward, traj_frames) in eval_results]
            rewards.append(np.mean(r))
        
        # Calculate mean and std dev
        median_reward = np.median(rewards)
        print(f"Median Reward:{median_reward:.2f}")

        # Log selection
        self.logger.log_selection(batch_index, 0, self.cumulative_exp, median_reward, 0)

        # Record individual mean results as well
        with open(self.result_file, mode='a') as f:
            row = [str(x) for x in rewards]
            row.insert(0, str(batch_index))

            row_str = ','.join(row)
            f.write(row_str + '\n')