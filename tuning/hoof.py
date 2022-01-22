import sys
import numpy as np

from tuning.utils import CalculateIS, OpeMemory, create_configs
from .base import TuningStrategy

class HOOF(TuningStrategy):
    def __init__(self, config, env, device, run_ideal):
       # Create configs and write to file
        alg_configs = create_configs(config['alg_name'], config['num_configs'], config['n_epochs'])

        super().__init__(config, env, device, alg_configs, run_ideal)

        # Initialize OPE memory
        self.ope_mem = OpeMemory(config['num_steps'] , config['ope_traj_len'])

        # Store some params on object
        self.num_iter = 0
        self.max_kl = config['max_kl']
        self.first_order = config['alg_name'] not in ['PPO']

    def update(self, batch_index, log=True):
        # Log batch info
        if self.is_on_policy:
            self.ope_mem.log_rollout_buffer(self.models[self.cfg_index].rollout_buffer, self.cfg_index)
        else:
            self.ope_mem.log_replay_buffer(self.models[self.cfg_index], self.cfg_index)

        # Cache original info
        orig_weights = self.models[self.cfg_index].policy.state_dict()
        orig_policy = self.models[self.cfg_index].policy

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

            # Train model
            if self.is_on_policy:
                self.models[j].train()
            else:
                self.models[j].train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

        # Update choice
        results = {}
        for j in range(self.num_configs):
            # Find probabilities of actions by evaluation policy
            D = self.ope_mem.get_ope_trajectories(self.models[j])

            # Calculate estimated average cumulative reward
            expected_reward = CalculateIS(D, True)

            # Calculate KL distance if first order method
            if self.first_order:
                kl_dist = self.ope_mem.calc_kl_dist(orig_policy, self.models[j])
            else:
                kl_dist = 0.0

            results[j] = (expected_reward, kl_dist)


        # Choose best algorithm based on estimated reward subject to kl constraint
        max_index = []
        max_reward = -sys.float_info.max
        for index, val in results.items():
            reward, kl_dist = val
            if kl_dist < self.max_kl:
                if reward > max_reward:
                    max_reward = reward
                    max_index = [index]
                elif reward == max_reward:
                    max_index.append(index)
        
        # If no KL distance is less than min, choose min kl distance
        min_kl_dist = sys.float_info.max
        if len(max_index) == 0:
            for index, val in results.items():
                _, kl_dist = val
                if kl_dist < min_kl_dist:
                    min_kl_dist = kl_dist
                    max_index = [index]
                elif kl_dist == min_kl_dist:
                    max_index.append(index)

        # Random tiebreak
        selected_index = np.random.choice(max_index)
        if log:
            self.cfg_index = selected_index

        # Reset memory
        self.ope_mem.reset()

        return selected_index

    def gather_data(self):
        self.num_iter += 1

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
        