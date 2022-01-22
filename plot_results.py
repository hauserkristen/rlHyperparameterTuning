import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from bisect import bisect_right

from utils import *

def main():
    # Create command line arguments
    parser = argparse.ArgumentParser(description='Plot results of hyperparameter tuning on an RL experiment.')
    parser.add_argument('variable', help='Environment or algorithm to plot results from.')
    parser.add_argument('-u', '--use_rolling_avg', help='Whether to plot maximum or rolling average.', action="store_true")
    parser.add_argument('-m', '--use_median', help='Whether to average across seeds or take the median.', action="store_true")
    parser.add_argument('-s', '--run_survey', help='Create plot for the survey.', action="store_true")
    parser.add_argument('-f', '--run_tuned', help='Create plot for the hand tuned.', action="store_true")
    parser.add_argument('-p', '--show_percentiles', help='Add percentile shading.', action="store_true")
    parser.add_argument('-c', '--use_color', help='Plot using colors.', action="store_true")
    args = parser.parse_args()

    # Constants
    algs = ['PPO', 'DQN', 'SAC']
    #algs = ['SAC']
    envs = ['CustomCartPole-v0', 'CustomLunarLander-v0', 'CustomReacher-v0', 'CustomHopper-v0']
    working_dir = 'results'

    # Font size
    FONT_SIZE = 18

    # Graph Unit
    GRAPH_UNIT = 7

    # Black-White styles
    dash_styles = [
        (0, ()), # Solid
        (0, (1, 1)), # Dotted
        (0, (5, 5)), # Dashed
        (0, (3, 5, 1, 5)), # Dash dotted
        (0, (5, 1)) # Densely dashed
    ] 
    data_dashes = {
        'HT-BOPS': dash_styles[0],
        'PBT': dash_styles[1],
        'HOOF': dash_styles[2],
        'Random Fixed': dash_styles[3],
        'Random Flex': dash_styles[3],
        'Hand Tuned': dash_styles[4]
    }
    
    # Color styles
    data_colors = {
        'HT-BOPS':  (0.0, 0.447, 0.741), # Blue
        'PBT': (0.466, 0.674, 0.188), # Green
        'HOOF':  (0.929, 0.694, 0.125), # Yellow
        'Random Fixed':  (0.494, 0.184, 0.556), # Purple
        'Random Flex': (0.635, 0.078, 0.184), # Maroon
        'Hand Tuned': (0.85, 0.32, 0.098), # Orange-Red
    }

    # Env name map
    env_names = {
        'CustomCartPole-v0': 'CartPole',
        'CustomLunarLander-v0': 'LunarLander',
        'CustomReacher-v0': 'Reacher',
        'CustomHopper-v0': 'Hopper'
    }

    # Solved 
    solved_rewards = {
        'CustomCartPole-v0': 195.0,
        'CustomLunarLander-v0': 20.0,
        'CustomReacher-v0': 11.0,
        'CustomHopper-v0': 3000.0
    }

    # Set reward bounds
    reward_y_graph_bounds = {
        'CustomCartPole-v0': (0, 210),
        'CustomLunarLander-v0': (-125, 25),
        'CustomReacher-v0': (0, 10),
        'CustomHopper-v0': (100, 2400)
    }

    reward_x_graph_bounds = {
        'CustomCartPole-v0': {
            'PPO': (-4e4, 4.5e5),
            'DQN': (-4e4, 4.5e5)
        },
        'CustomLunarLander-v0': {
            'PPO': (-4e4, 6.5e5),
            'DQN': (-2e5, 6.75e6)
        },
        'CustomReacher-v0': {
            'PPO': (-6e4, 1.1e6),
            'SAC': (-6e4, 1.1e6),
        },
        'CustomHopper-v0': {
            'PPO': (-2e5, 4.5e6),
            'SAC': (-2e5, 4.5e6)
        }
    }

    # Configure font sizes
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize

    # Identify if we should plot across env or alg
    if args.variable in algs:
        use_alg = True
        unused_list = envs
    elif args.variable in envs:
        use_alg = False
        unused_list = algs

    # Parse data
    item_data = {}
    for i_item in unused_list:
        if args.run_survey:
            if use_alg:
                data_dirs = {
                    '{}\\{}\\{}\\PBT\\'.format(working_dir, i_item, args.variable) : 'PBT',
                    '{}\\{}\\{}\\HOOF\\'.format(working_dir, i_item, args.variable) : 'HOOF',
                    '{}\\{}\\{}\\RANDOM_FIXED\\'.format(working_dir, i_item, args.variable) : 'Random Fixed',
                    '{}\\{}\\{}\\RANDOM_FLEX\\'.format(working_dir, i_item, args.variable) : 'Random Flex'
                }
            else:
                data_dirs = {
                    '{}\\{}\\{}\\PBT\\'.format(working_dir, args.variable, i_item) : 'PBT',
                    '{}\\{}\\{}\\HOOF\\'.format(working_dir, args.variable, i_item) : 'HOOF',
                    '{}\\{}\\{}\\RANDOM_FIXED\\'.format(working_dir, args.variable, i_item) : 'Random Fixed',
                    '{}\\{}\\{}\\RANDOM_FLEX\\'.format(working_dir, args.variable, i_item) : 'Random Flex'
                }
        elif args.run_tuned:
            if use_alg:
                data_dirs = {
                    '{}\\{}\\{}\\SEHOP\\'.format(working_dir, i_item, args.variable) : 'HT-BOPS',
                    '{}\\{}\\{}\\HOOF\\'.format(working_dir, i_item, args.variable) : 'HOOF',
                    '{}\\{}\\{}\\PBT\\'.format(working_dir, i_item, args.variable) : 'PBT',
                    '{}\\{}\\{}\\HAND_TUNED\\'.format(working_dir, i_item, args.variable) : 'Hand Tuned'
                }

            else:
                data_dirs = {
                    '{}\\{}\\{}\\SEHOP\\'.format(working_dir, args.variable, i_item) : 'HT-BOPS',
                    '{}\\{}\\{}\\HOOF\\'.format(working_dir, args.variable, i_item) : 'HOOF',
                    '{}\\{}\\{}\\PBT\\'.format(working_dir, args.variable, i_item) : 'PBT',
                    '{}\\{}\\{}\\HAND_TUNED\\'.format(working_dir, args.variable, i_item) : 'Hand Tuned'
                }
                
        elif use_alg:
            data_dirs = {
                '{}\\{}\\{}\\SEHOP\\'.format(working_dir, i_item, args.variable) : 'HT-BOPS',
                '{}\\{}\\{}\\HOOF\\'.format(working_dir, i_item, args.variable) : 'HOOF',
                '{}\\{}\\{}\\PBT\\'.format(working_dir, i_item, args.variable) : 'PBT',
                '{}\\{}\\{}\\RANDOM_FIXED\\'.format(working_dir, i_item, args.variable) : 'Random Fixed',
                '{}\\{}\\{}\\RANDOM_FLEX\\'.format(working_dir, i_item, args.variable) : 'Random Flex'
            }

        else:
            data_dirs = {
                '{}\\{}\\{}\\SEHOP\\'.format(working_dir, args.variable, i_item) : 'HT-BOPS',
                '{}\\{}\\{}\\HOOF\\'.format(working_dir, args.variable, i_item) : 'HOOF',
                '{}\\{}\\{}\\PBT\\'.format(working_dir, args.variable, i_item) : 'PBT',
                '{}\\{}\\{}\\RANDOM_FIXED\\'.format(working_dir, args.variable, i_item) : 'Random Fixed',
                '{}\\{}\\{}\\RANDOM_FLEX\\'.format(working_dir, args.variable, i_item) : 'Random Flex'
            }

        
        # Read data files in and create trace
        data_across_seeds = {}
        ordered_data_dirs = list(data_dirs.keys())
        ordered_data_dirs.sort()
        max_experience = 0
        for data_dir in ordered_data_dirs:
            label = data_dirs[data_dir]
            seed_data = {}

            # Find all seed directories
            seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir and 'Visualizations' not in x[0]]
            for seed_dir in seed_dirs:
                str_seed = seed_dir[-1]

                if label == 'Random Fixed':
                    experience, reward = read_random_result_file(seed_dir, args.use_rolling_avg)
                else:
                    experience, reward = read_result_file(seed_dir, args.use_rolling_avg)

                # Create trace
                seed_data[str_seed] = experience, reward

            # If no file found, skip
            if seed_data:
                # Identify max experience
                seed_max_experience = max([max(e) for e, r in seed_data.values()])
                if seed_max_experience > max_experience:
                    max_experience = seed_max_experience

                # Record
                data_across_seeds[label] = seed_data

        # Average across seeds
        data_traces = []
        min_e = None
        for label, seed_data in data_across_seeds.items():
            if use_alg:
                seed_data = extend_experience(seed_data, i_item, max_experience)
                x, y, min_y, max_y = average_across_trials(seed_data, args.use_median, i_item, max_experience)
            else:
                seed_data = extend_experience(seed_data, args.variable, max_experience)
                x, y, min_y, max_y = average_across_trials(seed_data, args.use_median, args.variable, max_experience)

            # Translate to positive rewards
            if np.min(y) < 0:
                offset = -np.min(y)
            else:
                offset = 0.0

            # Get environment name
            if use_alg: 
                env_name = i_item
            else:
                env_name = args.variable
            
            # Calculate various thresholds
            if not args.run_survey and not args.run_tuned:
                for threshold in [0.25, 0.5, 0.75, 0.90]:
                    threshold_index = bisect_right(y + offset, threshold * (solved_rewards[env_name] + offset) )
                    if threshold_index >= len(x):
                        print('{} {} {:e}'.format(threshold * (solved_rewards[env_name] + offset), np.max(y+offset), np.max(x)))
                        print('{}-{} Experience Required for {}: {}'.format(i_item, label, threshold, 'N/A'))
                    else:
                        print('{}-{} Experience Required for {}: {:e}'.format(i_item, label, threshold, x[threshold_index]))

                if min_e == None:
                    min_e = max(x)
                elif max(x) < min_e:
                    min_e = max(x)
                    
  
            data_traces.append((x, y, min_y, max_y, label))

        if len(data_traces) > 0:
            item_data[i_item] = data_traces

        # Print max
        if not args.run_survey and not args.run_tuned:
            for x,y, _, _, label in data_traces:
                max_index = bisect_right(x, min_e) 
                if max_index >= len(x):
                    print('{}-{} Max Reward: {}'.format(i_item, label, y[-1]))
                else:
                    print('{}-{} Max Reward: {}'.format(i_item, label, y[max_index]))

    # Create graph
    num_items = len(item_data.keys()) 
    if num_items == 1:
        fig = plt.figure(figsize=(GRAPH_UNIT * 1.4, GRAPH_UNIT))
        gs = gridspec.GridSpec(1, 1)
    elif num_items == 2:
        fig = plt.figure(figsize=(GRAPH_UNIT*2.4, GRAPH_UNIT))
        gs = gridspec.GridSpec(1, 4)
    else:
        fig = plt.figure(figsize=(GRAPH_UNIT*2, GRAPH_UNIT*2))
        gs = gridspec.GridSpec(2, 4)
        
    gs.update(wspace=1.1, hspace=0.3)

    axis_index = 0
    axes = []
    for i_item in unused_list:
        if i_item in item_data.keys():
            if num_items == 1:
                ax = fig.add_subplot(gs[0,axis_index])
            elif num_items == 2:
                ax = fig.add_subplot(gs[0, axis_index*2:axis_index*2+2])
            else:
                if axis_index == 0:
                    ax = fig.add_subplot(gs[0, :2])
                elif axis_index == 1:
                    ax = fig.add_subplot(gs[0, 2:])
                else:
                    ax = fig.add_subplot(gs[1, 1:3])

            axes.append(ax)

            for x, y, min_y, max_y, label in item_data[i_item]:
                if args.use_color:
                    ax.plot(x, y, color=data_colors[label], label=label)
                    if args.show_percentiles:
                        ax.fill_between(x, max_y, min_y, facecolor=data_colors[label], alpha=0.25)
                else:
                    ax.plot(x, y, color='black', linestyle=data_dashes[label], label=label)

            if i_item in env_names.keys():
                i_item = env_names[i_item]

            ax.set(xlabel='Experience', ylabel='Expected Reward')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.set_title(i_item)
            axis_index += 1

            # Set bounds
            if use_alg:
                ax.set_xlim(*reward_x_graph_bounds[i_item][args.variable])
                ax.set_ylim(*reward_y_graph_bounds[i_item])
            else:
                ax.set_xlim(*reward_x_graph_bounds[args.variable][i_item])
                ax.set_ylim(*reward_y_graph_bounds[args.variable])

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

            label = labels[i]
            unique_labels.append(label)
            

    if num_items == 1:
        box = axes[-1].get_position()
        axes[-1].set_position([box.x0+0.05, box.y0+0.05, box.width * 0.6, box.height*0.85])
    elif num_items == 2:
        box = axes[0].get_position()
        axes[0].set_position([box.x0, box.y0+0.05, box.width*0.85, box.height*0.875])

        box = axes[1].get_position()
        axes[1].set_position([box.x0-0.05, box.y0+0.05, box.width*0.85, box.height*0.875])
            
    axes[-1].legend(unique_lines, unique_labels, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)

    plt.show()

if __name__ == '__main__':
    main()