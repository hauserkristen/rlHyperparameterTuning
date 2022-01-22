import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

from utils import *

def read_variance_result_file(seed_directory):
    avg_result_file = os.path.join(seed_directory, 'alg_sel.csv')
    min_max_result_file = os.path.join(seed_directory, 'alg_result.csv')

    experience = []
    avg_rewards = []
    with open(avg_result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[2])
                rew_val = float(row_split[3])
                experience.append(exp_val)
                avg_rewards.append(rew_val)

            row_index += 1

    min_rewards = []
    max_rewards = []
    with open(min_max_result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                reward_vals = [float(v) for v in row_split[1:]]
                min_rewards.append(np.min(reward_vals))
                max_rewards.append(np.max(reward_vals))

            row_index += 1
            row_index += 1

    return experience, avg_rewards, min_rewards, max_rewards

def main():
    # Create command line arguments
    parser = argparse.ArgumentParser(description='Plot variance of results.')
    parser.add_argument('alg', help='Algorithm to plot results from.')
    parser.add_argument('env', help='Environment to plot results from.')
    parser.add_argument('seed', help='Seed to plot results from.')
    args = parser.parse_args()

    # Create path to directory
    data_dir = 'results\\{}\\{}\\RANDOM_FIXED\\'.format(args.env, args.alg)

    # Font size
    FONT_SIZE = 18

    # Configure font sizes
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize


    # Find seed directory
    seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]
    for seed_dir in seed_dirs:
        str_seed = seed_dir[-1]

        if str_seed == args.seed:
            # Read min and max reward
            experience, avg_reward, min_reward, max_reward = read_variance_result_file(seed_dir)

            # Calcualte rolling average
            roll_avg_reward = []
            roll_min_reward = []
            roll_max_reward = []
            for i in range(len(experience)):
                roll_avg_reward.append(np.mean(avg_reward[i-99:i+1]))
                roll_min_reward.append(np.mean(min_reward[i-99:i+1]))
                roll_max_reward.append(np.mean(max_reward[i-99:i+1]))

            # Create graph
            plt.fill_between(experience, roll_min_reward, roll_max_reward, color=(0.0, 0.447, 0.741), alpha=0.2)
            plt.plot(experience, roll_avg_reward, '--', color=(0.0, 0.447, 0.741))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            plt.xlabel('Experience')
            plt.ylabel('Expected Reward')
            plt.show()

if __name__ == '__main__':
    main()
