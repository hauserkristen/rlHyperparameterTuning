import math
import importlib
import numpy as np
from copy import deepcopy


try:
    from gym.envs.mujoco import HopperEnv as HopperBase
except:
    from pybullet_envs.gym_locomotion_envs import HopperBulletEnv as HopperBase

class CustomHopper(HopperBase):
    """
    Custom wrapper for Hopper, does not require modification since shaping is not used
    https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/roboschool/envs/locomotion/hopper_env.py
    """

    def __init__(self):
        self.current_traj_len = 0
        super(CustomHopper, self).__init__()
        
    def step(self, action, eval=False):
        self.current_traj_len += 1
        next_state, reward, done, info = super(CustomHopper, self).step(action)

        if self.current_traj_len == 1000:
            done = True

        return next_state, reward, done, info

    def reset(self):
        self.current_traj_len = 0
        return super(CustomHopper, self).reset()
