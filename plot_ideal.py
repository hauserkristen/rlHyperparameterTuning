import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import auc

from utils import *

def main():
    # Create command line arguments
    parser = argparse.ArgumentParser(description='Plot results of hyperparameter tuning on an RL experiment.')
    parser.add_argument('tune', help='Tuning algorithm to plot results from.')
    parser.add_argument('-c', '--use_color', help='Plot using colors.', action="store_true")
    parser.add_argument('-m', '--use_median', help='Whether to average across seeds or take the median.', action="store_true")
    args = parser.parse_args()

    # Constants
    working_dir = 'results'
    plot_orig = False

    # Experiments
    experiments = [
        ('A2C', 'CustomCartPole-v0'),
        ('PPO', 'CustomLunarLander-v0')
    ]

    # Font size
    FONT_SIZE = 18

    # Black-White styles
    primary_dash = (0, ()) # Solid
    secondary_dash = (0, (1, 1)) # Dotted
    original_dash = (0, (5, 5)), # Dashed

    # Color styles
    data_colors = {
        'HT-BOPS': (0.466, 0.674, 0.188),
        'HOOF': (0.635, 0.078, 0.184)
    }

    # Env name map
    env_names = {
        'CustomCartPole-v0': 'CartPole',
        'CustomLunarLander-v0': 'LunarLander',
        'CustomReacher-v0': 'Reacher'
    }

    # Configure font sizes
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize

    # Switch label name if SEHOP
    if args.tune == 'SEHOP':
        tuning_strat = 'HT-BOPS'
    else:
        tuning_strat = args.tune

    # Create graph
    GRAPH_UNIT = 5
    fig = plt.figure(figsize=(GRAPH_UNIT*2.8, 1.25*GRAPH_UNIT))
    gs = gridspec.GridSpec(1, 4)
    gs.update(wspace=1.1, hspace=0.3)

    # Loop through experiments
    axis_index = 0
    axes = []
    for alg, env in experiments:
        # Create directory to read from
        data_dir = '{}\\{}\\{}\\{}_IDEAL\\'.format(working_dir, env, alg, args.tune)
        primary_seed_data = {}
        secondary_seed_data = {}
        ideal_selection_data = {}

        # Find all seed directories for ideal
        max_experience = 0
        seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]
        for seed_dir in seed_dirs:
            str_seed = seed_dir[-1]

            # Read result file
            experience, primary_reward = read_result_file(seed_dir, False)

            # Read original alg selection
            primary_alg_sel =  read_alg_sel(seed_dir)

            # Read ideal file
            secondary_reward, secondary_alg_sel = read_ideal_result_file(seed_dir, False)
            secondary_experience = np.array(experience)

            # Adjust ideal file since first occurence is before experience
            secondary_experience = np.insert(secondary_experience, 0, 0)
            secondary_experience = secondary_experience[:-1]
            
            # Create trace
            primary_seed_data[str_seed] = experience, primary_reward
            secondary_seed_data[str_seed] = experience, secondary_reward

            # Get max experience
            if max(experience) > max_experience:
                max_experience = max(experience)

            # Calculate how many ideal choices were made
            ideal_selection_data[str_seed] = np.equal(primary_alg_sel, secondary_alg_sel)

        # Find original 
        if plot_orig:
            data_dir = '{}\\{}\\{}\\{}\\'.format(working_dir, args.env, args.alg, args.tune)
            seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]
            original_seed_data = {}
            for seed_dir in seed_dirs:
                str_seed = seed_dir[-1]

                # Read result file
                original_experience, original_reward = read_result_file(seed_dir, False)
                original_seed_data[str_seed] = original_experience, original_reward

                # Get max experience
                if max(original_experience) > max_experience:
                    max_experience = max(original_experience)

        # If no file found, skip
        if primary_seed_data and secondary_seed_data:
            # Average
            primary_x, primary_y, _, _ = average_across_trials(primary_seed_data, args.use_median, env, max_experience)
            secondary_x, secondary_y, _, _ = average_across_trials(secondary_seed_data, args.use_median, env, max_experience)
            if plot_orig and original_seed_data:
                original_x, original_y, _, _ = average_across_trials(original_seed_data, args.use_median, env, max_experience)

        # Truncate so curves are the same length
        if len(primary_x) < len(secondary_x):
            secondary_x = secondary_x[:len(primary_x)]
            secondary_y = secondary_y[:len(primary_x)]
        elif len(primary_x) > len(secondary_x):
            primary_x = primary_x[:len(secondary_x)]
            primary_y = primary_y[:len(secondary_x)]

        # Create subplot
        ax = fig.add_subplot(gs[0, axis_index*2:axis_index*2+2])

        # Identify color
        if args.use_color:
            color = data_colors[tuning_strat]
        else:
            color = 'black'

        # Plot
        ax.plot(primary_x, primary_y, color=color, linestyle=primary_dash, label=tuning_strat)
        ax.plot(secondary_x, secondary_y, color=color, linestyle=secondary_dash, label='{} IDEAL'.format(tuning_strat))
        if plot_orig and original_seed_data:
            ax.plot(original_x, original_y, color=color, linestyle=original_dash, label='Original {}'.format(tuning_strat))
            
        ax.set(xlabel='Experience', ylabel='Expected Reward')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.set_title('{} on {}'.format(alg, env_names[env]))

        # Update
        axes.append(ax)
        axis_index += 1

        # Make AUC calculation positive
        if np.min(primary_y) < 0 or np.min(secondary_y) < 0:
            offset = np.max([-np.min(primary_y), -np.min(secondary_y)])
            pos_primary_y = primary_y + offset
            pos_secondary_y = secondary_y + offset
        else:
            pos_primary_y = primary_y
            pos_secondary_y = secondary_y 

        # Calculate AUC
        original_auc = auc(primary_x, pos_primary_y)
        ideal_auc = auc(secondary_x, pos_secondary_y)
        area_between = original_auc / ideal_auc
        print('{}-{} Regret: {}'.format(alg, env, area_between))

        # Calculate the ideal choice percent
        total_num_choices = np.sum([len(ideal_sel) for ideal_sel in ideal_selection_data.values()])
        ideal_choice_made = float(np.sum([np.sum(ideal_sel) for ideal_sel in ideal_selection_data.values()]))
        print('{}-{} Percent Ideal Choices: {}'.format(alg, env, ideal_choice_made /  total_num_choices))

    # One legend
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # Remove duplicates
    seen = set()
    unique_indices = [i for i, x in enumerate(labels) if x not in seen and not seen.add(x)]
    unique_lines = []
    unique_labels = []
    for i in range(len(lines)):
        if i in unique_indices:
            unique_lines.append(lines[i])
            unique_labels.append(labels[i])

    # Position legend
    axes[-1].legend(unique_lines, unique_labels, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    # Resize plots to fit legend
    box = axes[0].get_position()
    axes[0].set_position([box.x0-0.025, box.y0+0.05, box.width*0.85, box.height*0.875])
    box = axes[1].get_position()
    axes[1].set_position([box.x0-0.075, box.y0+0.05, box.width*0.85, box.height*0.875])

    # Plot
    plt.show()


if __name__ == '__main__':
    main()