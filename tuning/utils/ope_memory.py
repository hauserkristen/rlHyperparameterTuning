import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from gym import spaces
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.distributions import sum_independent_dims
from stable_baselines3 import SAC

class ReplayTrajectory(object):
    def __init__(self, data, ope_traj_length: int):
        self.max_traj_length = ope_traj_length
        self.done = False

        if isinstance(data, dict):
            self._records = data
        else:
            self._records = dict([(key, []) for key in data])

        self._record_lengths = {k:len(v) for k,v in self._records.items()}

    def set_done(self, value: bool):
        self.done = value

    def add_record(self, key: str, data: list):
        if len(data) > self.max_traj_length and self.max_traj_length != -1:
            raise Exception('Replay Trajectory record too large given maximum trajectory length.')

        self._records[key] = data
        self._record_lengths[key] = len(data)

    def add_value(self, key: str, value: float):
        self._records[key].append(value)

        if len(self._records[key]) > self.max_traj_length and self.max_traj_length != -1:
            raise Exception('Replay Trajectory already at maximum capacity.')
        
        self._record_lengths[key] = len(self._records[key])

    def get_values(self, key: str):
        return np.array(self._records[key])

    def __len__(self):
        return min(self._record_lengths.values())

class OpeMemory(object):
    def __init__(self, memory_size: int, ope_traj_length: int = -1):
        self.memory_size = memory_size
        self.ope_traj_length = ope_traj_length
        self.memory_index = -1
        self.full_memory = False
        self.behavior_memory = []
        self.behavior_alg_indices = []
        self.behavioral_keys = ['States', 'Actions', 'Rewards', 'Action Log Probabilities']
        self.ope_keys = ['Behavior Log Probabilities', 'Behavior Rewards', 'Evaluation Log Probabilities']
        self.last_removed_alg_index = -1

    def check_done(self, traj):
        if self.ope_traj_length == -1:
            return traj.done
        else:
            return len(traj) == self.ope_traj_length

    def get_window_size(self):
        return min(self.memory_size, len(self.behavior_memory))

    def get_ope_trajectories(self, model):
        D = []
        for behav_traj in self.behavior_memory:
            if self.check_done(behav_traj):
                ope_traj = ReplayTrajectory(self.ope_keys, self.ope_traj_length)

                # Copy records
                ope_traj.add_record(self.ope_keys[0], behav_traj.get_values(self.behavioral_keys[3]).flatten())
                ope_traj.add_record(self.ope_keys[1], behav_traj.get_values(self.behavioral_keys[2]).flatten())

                # Get states and actions, then action probabilties
                behav_states = behav_traj.get_values('States')
                behav_actions = behav_traj.get_values('Actions')
                if issubclass(type(model), OnPolicyAlgorithm):
                    with torch.no_grad():
                        # Convert to pytorch tensor
                        obs_tensor = torch.as_tensor(behav_states).to(model.device)
                        act_tensor = torch.as_tensor(behav_actions).to(model.device)
                        if isinstance(model.action_space, spaces.Discrete):
                            act_tensor = act_tensor.long().flatten()
                        
                        # Get action log prob
                        _, log_probs, _ = model.policy.evaluate_actions(obs_tensor, act_tensor)

                        # Convert from GPU if necessary
                        if log_probs.is_cuda:
                            log_probs = log_probs.cpu()
                        log_probs = log_probs.numpy()
                elif isinstance(model, SAC):
                    with torch.no_grad():
                        # Convert to pytorch tensor
                        obs_tensor = torch.as_tensor(behav_states).to(model.device)
                        act_tensor = torch.as_tensor(behav_actions).to(model.device)

                        # Get action log prob
                        mean_actions, log_std, kwargs = model.actor.get_action_dist_params(obs_tensor)
                        std_actions = torch.ones_like(mean_actions) * log_std.exp()
                        norm_dist = Normal(mean_actions, std_actions)
                        log_probs = sum_independent_dims(norm_dist.log_prob(act_tensor))

                        # Convert from GPU if necessary
                        if log_probs.is_cuda:
                            log_probs = log_probs.cpu()
                        log_probs = log_probs.numpy()
                else:
                    traj_len = len(behav_traj)
                    log_probs = np.zeros((traj_len))

                    # If model is unused, initialize rate
                    if model.exploration_rate == 0:
                        exploration_rate = model.exploration_schedule(model._current_progress_remaining)
                    else:
                        exploration_rate = model.exploration_rate

                    non_greedy_prob = exploration_rate / model.action_space.n

                    # Get log probability since it is not stored in replay buffer
                    for i in range(traj_len):
                        with torch.no_grad():
                            # Convert to pytorch tensor
                            obs_tensor = torch.as_tensor(behav_states[i]).to(model.device)

                            # Check if action was greedy
                            greedy_action, _ = model.predict(obs_tensor, deterministic=True)

                            # Convert from GPU if necessary
                            if greedy_action.is_cuda:
                                greedy_action = greedy_action.cpu()
                            greedy_action = greedy_action.numpy()

                            if np.array_equal(behav_actions[i].flatten(), greedy_action):
                                log_probs[i] = np.log(non_greedy_prob + (1-exploration_rate))
                            else:
                                log_probs[i] = np.log(non_greedy_prob)

                # Get taken action probabiltlies
                ope_traj.add_record(self.ope_keys[2], log_probs)

                D.append(ope_traj)

        return D

    def calc_kl_dist(self, behav_policy, eval_model):
        kl_dists = []
        for behav_traj in self.behavior_memory:
            if behav_traj.done or len(behav_traj) == self.ope_traj_length:

                # Get states then action probabilty distribution
                behav_states = behav_traj.get_values('States')
                with torch.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = torch.as_tensor(behav_states).to(eval_model.device)

                    if isinstance(eval_model, SAC):
                        behav_mean_actions, behav_log_std, _ = behav_policy.actor.get_action_dist_params(obs_tensor)
                        behav_std_actions = torch.ones_like(behav_mean_actions) * behav_log_std.exp()
                        behav_norm_dist = Normal(behav_mean_actions, behav_std_actions)

                        eval_mean_actions, eval_log_std, _ = eval_model.policy.actor.get_action_dist_params(obs_tensor)
                        eval_std_actions = torch.ones_like(eval_mean_actions) * eval_log_std.exp()
                        eval_norm_dist = Normal(eval_mean_actions, eval_std_actions)

                        kl_dist = kl_divergence(behav_norm_dist, eval_norm_dist).detach()

                        if kl_dist.is_cuda:
                            kl_dist = kl_dist.cpu()

                        kl_dist = kl_dist.numpy()
                    else:
                        # Get behav distribution
                        latent_pi, _, latent_sde = behav_policy._get_latent(obs_tensor)
                        behav_dist = behav_policy._get_action_dist_from_latent(latent_pi, latent_sde)

                        latent_pi, _, latent_sde = eval_model.policy._get_latent(obs_tensor)
                        eval_dist = eval_model.policy._get_action_dist_from_latent(latent_pi, latent_sde)

                        kl_dist = kl_divergence(behav_dist.distribution, eval_dist.distribution).detach()

                        if kl_dist.is_cuda:
                            kl_dist = kl_dist.cpu()

                        kl_dist = kl_dist.numpy()

                kl_dists.append(np.nanmean(kl_dist))

        return np.nanmean(kl_dists)

    def log_step(self, state, action, reward, done, log_prob, behav_alg_index):
        # Check for previous completed trajectory
        if self.memory_index == -1:
            self.__create_new_replay_traj(behav_alg_index)
        elif self.check_done(self.behavior_memory[self.memory_index]):
            self.__create_new_replay_traj(behav_alg_index)
            self.full_memory = False

        # Store records
        self.behavior_memory[self.memory_index].add_value(self.behavioral_keys[0], state.flatten())
        self.behavior_memory[self.memory_index].add_value(self.behavioral_keys[1], action.flatten())
        self.behavior_memory[self.memory_index].add_value(self.behavioral_keys[2], reward)
        self.behavior_memory[self.memory_index].add_value(self.behavioral_keys[3], log_prob)

        # Set done
        done = done.flatten()[0] == 1.0
        self.behavior_memory[self.memory_index].set_done(done)

        # Update if memory is full
        if len(self.behavior_memory) == self.memory_size and self.check_done(self.behavior_memory[self.memory_index]):
            self.full_memory = True

        # TODO: DEBUG
        traj_log_p = [np.sum(t.get_values('Action Log Probabilities')) for t in self.behavior_memory]
        num_zero = len(self.behavior_memory) - np.count_nonzero(traj_log_p)
        if done and num_zero > 0:
            print('Num Zero Trajs: {}'.format(num_zero))

    def log_rollout_buffer(self, rollout_buffer, behav_alg_index: int):
        for i in range(rollout_buffer.buffer_size):
            self.log_step(rollout_buffer.observations[i], rollout_buffer.actions[i], rollout_buffer.rewards[i], rollout_buffer.dones[i], rollout_buffer.log_probs[i], behav_alg_index)

    def log_replay_buffer(self, model, behav_alg_index: int):
        # Identify indicies of new steps
        if model.replay_buffer.pos - model.train_freq < 0:
            # Identify newest index
            if model.replay_buffer.pos - 1 < 0:
                current_index = model.replay_buffer.buffer_size - 1
            else:
                current_index = model.replay_buffer.pos - 1

            # Loop until number of new steps have been recorded
            new_step_indices = []
            while len(new_step_indices) < model.train_freq:
                # Add current index
                new_step_indices.append(current_index)

                # Adjust current index
                if model.replay_buffer.pos - 1 < 0:
                    current_index = model.replay_buffer.buffer_size - 1
                else:
                    current_index -= 1

        else:
            new_step_indices = range(model.replay_buffer.pos - model.train_freq, model.replay_buffer.pos)

        # Log steps
        for i in new_step_indices:
            # Get lot probability since it is not stored in replay buffer
            with torch.no_grad():
                # Convert to pytorch tensor
                obs_tensor = torch.as_tensor(model.replay_buffer.observations[i]).to(model.device)

                if isinstance(model, SAC):
                    _, log_probs = model.actor.action_log_prob(obs_tensor)
                    if log_probs.is_cuda:
                        log_probs = log_probs.cpu()
                    log_probs = log_probs.numpy()
                else:
                    # Check if action was greedy
                    greedy_action, _ = model.predict(obs_tensor)

                    # Calculate probability
                    non_greedy_prob = model.exploration_rate / model.action_space.n 
                    if np.array_equal(model.replay_buffer.actions[i], greedy_action):
                        log_probs = np.log(non_greedy_prob + (1-model.exploration_rate))
                    else:
                        log_probs = np.log(non_greedy_prob)

            self.log_step(model.replay_buffer.observations[i], model.replay_buffer.actions[i], model.replay_buffer.rewards[i], model.replay_buffer.dones[i], log_probs, behav_alg_index)

    def __create_new_replay_traj(self, alg_index: int):
        traj = ReplayTrajectory(self.behavioral_keys, self.ope_traj_length)

        # Cycle memory index
        if self.memory_index == self.memory_size - 1:
            self.memory_index = 0
        else:
            self.memory_index += 1

        # Add or replace trajectory
        if len(self.behavior_memory) < self.memory_size:
            self.behavior_alg_indices.append(alg_index)
            self.behavior_memory.append(traj)
        else:
            # Record replaced trajectory owner
            self.last_removed_alg_index = self.behavior_alg_indices[self.memory_index]

            # Replace
            self.behavior_memory[self.memory_index] = traj
            self.behavior_alg_indices[self.memory_index] = alg_index

    def reset(self):
        self.last_removed_alg_index = -1
        self.memory_index = -1
        self.behavior_memory = []
        self.behavior_alg_indices = []
        self.full_memory = False