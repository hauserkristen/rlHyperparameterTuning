import os
import numpy as np
from collections import deque

class UCBValue(object):
    def __init__(self, use_arm_variance: bool, use_estimate_variance: bool):
        self.expected_value = 0
        self.variance = 0
        self.use_count = 0
        self.use_arm_variance = use_arm_variance
        self.use_estimate_variance = use_estimate_variance

    def log(self, expected_value: float, variance: float):
        self.expected_value = expected_value
        self.variance = variance

    def calculate(self, min_reward: float, max_reward: float, p: int, window_size: int) -> float:
        if self.use_count == 0:
            return np.inf
        else:
            variance = 0.0

            if self.use_arm_variance:
                reward_bound = np.power(max_reward - min_reward, 2)
                variance += (reward_bound * np.log(p * window_size)) / (-2 * self.use_count)
                
            if self.use_estimate_variance:
                variance += self.variance

            if variance > 0:
                return self.expected_value + np.sqrt(variance)
            else:
                return self.expected_value
    
    def update_use_count(self, removed=False):
        if removed:
            if self.use_count > 0:
                self.use_count -= 1
        else:
            self.use_count += 1

    def reset(self):
        self.expected_value = 0
        self.variance = 0
        self.use_count = 0

class UCB(object):
    def __init__(self, bounding_prob: float, min_reward: float, max_reward: float, num_arms: int, use_rand_tiebreak: bool, use_arm_variance: bool, use_estimate_variance: bool):
        self.arms = [UCBValue(use_arm_variance, use_estimate_variance) for x in range(num_arms)]
        self.use_rand_tiebreak = use_rand_tiebreak
        self.p = bounding_prob

        # Calculate min and max reward in log space
        self.min_reward = min_reward
        self.max_reward = max_reward

    def _calculate_arms(self, window_size: int):
        # Calculate values
        ucb_payoff = [0.0] * len(self.arms)
        for i_arm, arm in enumerate(self.arms):
            ucb_payoff[i_arm] = arm.calculate(self.min_reward, self.max_reward, self.p, window_size)

        return ucb_payoff

    def calculate(self, episode_num: int, current_index: int, window_size: int):
        ucb_payoff = self._calculate_arms(window_size)

        # Choose max value
        if np.inf in ucb_payoff:
            unchoosen_arms = [i for i, e in enumerate(ucb_payoff) if e == np.inf]
            chosen_index = np.random.choice(unchoosen_arms)
        else:
            # Find all maximum
            max_ucb = np.max(ucb_payoff)
            max_indices = [i for i,x in enumerate(ucb_payoff) if x == max_ucb]

            # If current index in tie, choose that
            if not self.use_rand_tiebreak and current_index in max_indices:
                chosen_index = current_index
            else:
                # Random tie break
                chosen_index = np.random.choice(max_indices)

        return chosen_index

    def used_all_arms(self):
        used = [a.use_count > 0 for a in self.arms]
        return np.all(used)

    def get_indices(self, count: int, window_size: int):
        ucb_payoff = self._calculate_arms(window_size)

        if np.inf in ucb_payoff:
            return [], []
        else:
            arg_sorted = np.array(ucb_payoff).argsort()
            return arg_sorted[-count:], arg_sorted[:count]

    def log(self, i_arm: int, expected_value: float, variance: float):
        self.arms[i_arm].log(expected_value, variance)

    def get_arm_value(self, i_arm: int):
        arm = self.arms[i_arm]
        return (arm.expected_value, arm.variance)

    def reset(self):
        for arm in self.arms:
            arm.reset()