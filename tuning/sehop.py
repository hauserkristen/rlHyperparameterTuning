import gym
import torch
import numpy as np
from gym import spaces
from copy import deepcopy
from torch.distributions import Normal
from stable_baselines3 import SAC
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.distributions import sum_independent_dims

from .base import TuningStrategy
from tuning.utils import CalculateSIRPF, CalculateIS, OpeMemory,  UCB, create_configs

class SEHOP(TuningStrategy):
    def __init__(self, config, env, device, run_ideal):
        # Create configs and write to file
        alg_configs = create_configs(config['alg_name'], config['num_configs'], config['n_epochs'])

        super().__init__(config, env, device, alg_configs, run_ideal)

        # Initialize OPE memory
        self.ope_mem = OpeMemory(config['mem_size'])

        # Initialize UCB
        self.ucb_calc = UCB(config['bounding_prob'], *config['reward_bounds'], config['num_configs'], True, True, True)

        # Store some params on object
        self.min_reward = config['reward_bounds'][0]
        self.quantile_count = int(np.ceil(self.num_configs * 0.2))
        self.num_iter = 0
        self.exp_iter = config['exp_iter']

    def __evaluate(self, model):
        experience = 0
        done = False
        obs = self.eval_env.reset()
        while not done:
            # Predict
            if self.is_on_policy:
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
            next_obs, reward, done, _ = self.eval_env.step(action, True)

            # Get log probability
            if issubclass(type(model), OnPolicyAlgorithm):
                with torch.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = torch.as_tensor(obs).to(model.device)
                    act_tensor = torch.as_tensor(action).to(model.device)
                    if isinstance(model.action_space, spaces.Discrete):
                        act_tensor = act_tensor.long()

                    # Add dimension
                    obs_tensor = obs_tensor.reshape(1,*obs_tensor.shape)
                    act_tensor = act_tensor.reshape(1,*act_tensor.shape)

                    _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, act_tensor)

                    if log_prob.is_cuda:
                            log_prob = log_prob.cpu()

                    # Product across actions
                    log_prob = log_prob.numpy()
            elif isinstance(model, SAC):
                with torch.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = torch.as_tensor(obs).to(model.device)
                    act_tensor = torch.as_tensor(action).to(model.device)

                    # Get action log prob
                    mean_actions, log_std, kwargs = model.actor.get_action_dist_params(obs_tensor)
                    std_actions = torch.ones_like(mean_actions) * log_std.exp()
                    norm_dist = Normal(mean_actions, std_actions)
                    log_prob = sum_independent_dims(norm_dist.log_prob(act_tensor))

                    if log_prob.is_cuda:
                            log_prob = log_prob.cpu()

                    log_prob = log_prob.numpy()
                    
            else:
                #TODO: Fix for continuous
                non_greedy_prob = model.exploration_rate / model.action_space.n 
                log_prob = np.log(non_greedy_prob + (1-model.exploration_rate))

            # Add dimension to log
            obs_log = obs.reshape(1, *obs.shape)
            reward_log = np.array([reward])
            action_log = action_log.reshape(1, *action_log.shape)
            done_log = np.array([done])

            # Log step
            self.ope_mem.log_step(obs_log, action_log, reward_log, done_log, log_prob, self.cfg_index)

            # Update observation
            obs = next_obs

            experience += 1

        # If traj was replaced in OPE memory, remove use count from UCB
        if self.ope_mem.last_removed_alg_index != -1:
            self.ucb_calc.arms[self.ope_mem.last_removed_alg_index].update_use_count(True)

        return experience

    def update(self, batch_index, log=True):
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

            # Train model
            if self.is_on_policy:
                self.models[j].train()
            else:
                self.models[j].train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

        # Record existing memory if not log
        if not log:
            existing_ope_mem = deepcopy(self.ope_mem)
            existing_ucb_calc = deepcopy(self.ucb_calc)
            existing_cumulative_exp = deepcopy(self.cumulative_exp)

        if self.num_iter > 1:
            # Run evaluation trajectory and record in OPE memory
            self.cumulative_exp += self.__evaluate(self.models[self.cfg_index])

            # Calculate OPE
            for j in range(self.num_configs):
                # Find probabilities of actions by evaluation policy
                D = self.ope_mem.get_ope_trajectories(self.models[j])

                # Calculate estimated average cumulative reward
                self.ucb_calc.log(j, *CalculateSIRPF(D, self.min_reward))

            # Choose best algorithm based on UCB, after every arm has been pulled once
            selected_index = self.ucb_calc.calculate(batch_index, self.cfg_index, self.ope_mem.get_window_size())
        else:
            # Create temporary OPE memory
            temp_ope_memory = OpeMemory(500, 16)

            # Log training data
            if self.is_on_policy:
                temp_ope_memory.log_rollout_buffer(self.models[self.cfg_index].rollout_buffer, self.cfg_index)
            else:
                temp_ope_memory.log_replay_buffer(self.models[self.cfg_index], self.cfg_index)

            results = []
            for j in range(self.num_configs):
                # Find probabilities of actions by evaluation policy
                D = temp_ope_memory.get_ope_trajectories(self.models[j])

                # Calculate estimated average cumulative reward
                expected_reward = CalculateIS(D, True)

                # Set reward, variance and use count
                self.ucb_calc.arms[j].expected_value = expected_reward
                self.ucb_calc.arms[j].use_count = 1
                self.ucb_calc.arms[j].variance = 0.0

                # Record reward
                results.append(expected_reward)

            # Choose argmax
            selected_index = np.argmax(results)

        # Check for exploitation
        if log and self.num_iter % self.exp_iter == 0:
            upper_quantile, lower_quantile = self.ucb_calc.get_indices(self.quantile_count, self.ope_mem.get_window_size())
            if len(upper_quantile) > 0 and len(lower_quantile) > 0: 
                for j in lower_quantile:
                    # Get random upper quantile index
                    copy_index = np.random.choice(upper_quantile)
                    pytorch_state_dict = self.models[copy_index].policy.state_dict()

                    # Replace FA values
                    self.models[j].policy.load_state_dict(pytorch_state_dict)

                    # Reset use count
                    self.ucb_calc.arms[j].use_count = 0

        # Update selected index if log
        if log:
            self.cfg_index = selected_index
        else:
            self.ope_mem = existing_ope_mem
            self.ucb_calc = existing_ucb_calc
            self.cumulative_exp = existing_cumulative_exp

        return selected_index

    def gather_data(self):
        self.num_iter += 1

        # Collect data
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

        # Record uses
        self.ucb_calc.arms[self.cfg_index].update_use_count()

