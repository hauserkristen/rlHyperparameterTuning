import math
import numpy as np
from copy import deepcopy
from pybullet_envs.gym_manipulator_envs import ReacherBulletEnv

class CustomReacher(ReacherBulletEnv):
    """
    Custom wrapper for Reacher, does not require modification since shaping is not used
    https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/roboschool/envs/manipulation/reacher_env.py
    """

    def __init__(self):
        self.current_traj_len = 0
        super(CustomReacher, self).__init__()
        
    def step(self, action, eval=False):
        self.current_traj_len += 1
        next_state, reward, done, info = super(CustomReacher, self).step(action)

        if self.current_traj_len == 150:
            done = True

        return next_state, reward, done, info

    def reset(self):
        self.current_traj_len = 0
        return super(CustomReacher, self).reset()
