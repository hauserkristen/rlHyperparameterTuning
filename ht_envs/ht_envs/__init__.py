from gym.envs.registration import register

register(
    id='CustomLunarLander-v0',
    entry_point='ht_envs.envs:CustomLunarLander',
)

register(
    id='CustomContinuousLunarLander-v0',
    entry_point='ht_envs.envs:CustomContinuousLunarLander',
)

register(
    id='CustomCartPole-v0',
    entry_point='ht_envs.envs:CustomCartPole',
)

register(
    id='CustomReacher-v0',
    entry_point='ht_envs.envs:CustomReacher',
    reward_threshold=18.0
)

register(
    id='CustomHopper-v0',
    entry_point='ht_envs.envs:CustomHopper',
)