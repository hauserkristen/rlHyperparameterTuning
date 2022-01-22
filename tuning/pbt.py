import gym
import numpy as np
from gym import spaces

from tuning.utils import create_configs
from .base import TuningStrategy

class PBT(TuningStrategy):
    def __init__(self, config, env, device):
        # Create configs and write to file
        configs = create_configs(config['alg_name'], config['exp_num_configs'], config['n_epochs'])
        super().__init__(config, env, device, configs, False)

        # Select subset
        self.cfg_index_subset = np.random.choice(config['exp_num_configs'], config['num_configs'], replace=False)

        # Store some params on object
        self.quantile_count = int(np.ceil(self.num_configs * 0.2))
        self.num_iter = 0
        self.exp_iter = config['exp_iter']

    def __evaluate(self, model, num_episodes):
        episode_cumulative_rewards = []
        experience = 0
        for i in range(num_episodes):
            episode_reward = 0.0
            obs = self.eval_env.reset()
            done = False
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

                # Update observation and experience
                obs = next_obs
                experience += 1

                # Update reward
                episode_reward += reward

            episode_cumulative_rewards.append(episode_reward)

        # Calculate mean reward
        mean_reward = np.mean(episode_cumulative_rewards)

        return mean_reward, experience

    def update(self, batch_index, log=True):
        # Train models
        rewards = []
        for j in self.cfg_index_subset:
            # Train model
            if self.is_on_policy:
                self.models[j].train()
            else:
                self.models[j].train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

            # Evaluate
            mean_reward, experience = self.__evaluate(self.models[j], 10)
            rewards.append(mean_reward)

            # Record experience
            if log:
                self.cumulative_exp += experience

        # Only calculate next iteration if log is set
        if log:
            # Identify top and 
            self.cfg_index = np.argmax(rewards)

            # Check for exploitation
            if self.num_iter > self.exp_iter:
                upper_quantile = np.argpartition(rewards, -self.quantile_count)[self.quantile_count:]
                lower_quantile = np.argpartition(rewards, self.quantile_count)[:self.quantile_count]
                for j in lower_quantile:
                    # Get actual indicies
                    lower_quantile_index = self.cfg_index_subset[j]

                    # Get random upper quantile index
                    copy_index = self.cfg_index_subset[np.random.choice(upper_quantile)]
                    pytorch_state_dict = self.models[copy_index].policy.state_dict()

                    # Get random unused config
                    unused_cfgs = list((set(np.arange(self.num_configs)) - set(self.cfg_index_subset)) | set([lower_quantile_index]))
                    cfg_index = np.random.choice(unused_cfgs)
                    self.cfg_index_subset[j] = cfg_index

                    # Replace FA values
                    self.models[j].policy.load_state_dict(pytorch_state_dict)

                    # TODO: Log changes in configs

                    # Reset experience
                    self.num_iter = 0

    def gather_data(self):
        # Collect rollouts for all configurations
        for i in self.cfg_index_subset:
            if self.is_on_policy:
                self.models[i].collect_rollouts(
                    self.models[i].env, 
                    self.callbacks[i], 
                    self.models[i].rollout_buffer, 
                    self.num_steps
                )

                # Record experience
                self.cumulative_exp += self.num_steps
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
                self.cumulative_exp += self.models[i].train_freq

        self.num_iter += 1
    