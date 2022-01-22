from gym.envs.classic_control import CartPoleEnv
import numpy as np
from copy import deepcopy

class CustomCartPole(CartPoleEnv):
    """
    Custom wrapper for CartPole, does not require modification since shaping is not used
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        self.current_traj_len = 0
        super(CustomCartPole, self).__init__()
        
    def step(self, action, eval=False):
        self.current_traj_len += 1
        next_state, reward, done, info = super(CustomCartPole, self).step(action)

        if self.current_traj_len == 200:
            done = True

        return next_state, reward, done, info

    def reset(self):
        self.current_traj_len = 0
        return super(CustomCartPole, self).reset()
