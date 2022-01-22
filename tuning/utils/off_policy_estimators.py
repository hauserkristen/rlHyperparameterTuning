import os
from copy import deepcopy
import numpy as np
from scipy.special import logsumexp

from .ope_memory import ReplayTrajectory

EPSILON = 1e-20

def CalculateRewardLogProb(B):
    # Get the number of trajectories
    n = len(B)

    # Importance weights and returns
    P = np.zeros((n), dtype=np.float64)
    for i in range(n):
        P[i] = np.sum(B[i].get_values('Evaluation Log Probabilities'))

    return P

def CalculateLogIW(B):
    # Get the number of trajectories
    n = len(B)

    # Log probabilities and returns
    log_IWs = np.zeros((n), dtype=float)
    R = np.zeros((n), dtype=float)
    for i in range(n):
        e_prob = B[i].get_values('Evaluation Log Probabilities')
        b_prob = B[i].get_values('Behavior Log Probabilities')

        log_IWs[i] = np.sum(e_prob) - np.sum(b_prob)
        R[i] = np.sum(B[i].get_values('Behavior Rewards'))

    return log_IWs, R

"""
Importance sampling (IS) estimator. This can be weighted or unweighted.
"""
def CalculateIS(B, weighted_sampling: bool):
    # Get the number of trajectories
    n = len(B)

    # Importance weights and returns
    log_IWs, R = CalculateLogIW(B)

    # Enforce strictly positive reward, adding same value for each iteration since B is the same data set
    if min(R) <= 0:
        min_val = np.abs(min(R))
        R = np.add(R, min_val)
    R = np.add(R, EPSILON)

    # Reward positive check
    if np.any(R <= 0):
        print('Negative reward of: {}'.format(np.min(R)))
        input()

    # Calcualte weighted returns in log space
    weighted_R = logsumexp(log_IWs + np.log(R))

    # Mean
    if weighted_sampling:
        expected_log_reward = weighted_R - logsumexp(log_IWs)
    else:
        expected_log_reward = weighted_R - np.log(n)

    return expected_log_reward

"""
Sample Importance Resample (SIR) Particle filter (PF) estimator.
"""
def CalculateSIRPF(B, min_reward: float):
    # Importance weights and returns
    log_IWs, R = CalculateLogIW(B)

    # Resample using Gumbel-Max trick to sample of unnormalzied log probability distribution
    num_particles = len(B)
    gumbel_samples = np.random.gumbel(size=(num_particles, num_particles))
    resampled_R = R[np.argmax(log_IWs + gumbel_samples, axis=1)]

    # Estiamte mean and variance in log space
    uniform_expected_reward= np.mean(resampled_R)
    uniform_variance_reward= np.var(resampled_R) * np.sqrt(num_particles) / float(num_particles)
    
    return uniform_expected_reward, uniform_variance_reward
