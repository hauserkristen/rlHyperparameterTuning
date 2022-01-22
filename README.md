# rltuning

This project contains tuning methods for Reinforcement Learning algorithms. The tuning directory contains 3 methods: 
* Population Based Training (PBT)
* Hyperparameter Optimization on the Fly (HOOF)
* Hyperparameter Tuning using Bandits and Off-Policy Sampling (HT-BOPS)

Additionally, it uses openbaselines3 which utilizes Pytorch for the learning algorithm implementations. The algorithms used within this work were:
* Deep Q-Network (DQN)
* Advantage Actor Critic (A2C)
* Proximal Policy Optimization (PPO)
* Soft Afector Critic (SAC)
Other algorithms exist within the openbaselines3 implemenentations and could be integrated by adjusting the tuning base class.

Additionally, environments are slightly modified to exclude any shaping functions during evaluation trajectories. To install these environments navigate to the ht_envs directory and run the command: `pip install -e .`

The main tuning file is `main.py` and requires at minimum four parameters:
* seed - Random seed for the experiment
* tuning_strategy_name - Tuning strategy to use, which can be: SEHOP, HOOF, PBT
* alg_name - Algorithm to tune, which can be: PPO, A2C, DQN, SAC
* env_name - Environment to tune on, which can be: CustomLunarLander-v0, CustomCartPole-v0 CustomReacher-v0, or CustomHopper-v0

Optional parameters include:
* -v or --visualize - Visualize and save evaluation trajectories. Default=False
* -s or --save - Save model. Default=False
* -l or --load - Loads previous save point. Default=False