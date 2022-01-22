import os
import numpy as np
from bisect import bisect

def read_random_result_file(result_dir: str, use_rolling_avg: bool):
    # Record experience
    experience = []
    result_file = os.path.join(result_dir, 'alg_sel.csv')
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[2])
                experience.append(exp_val)

            row_index += 1

    # Record reward per index
    reported_rewards = {}
    reward_history = {}
    result_file = os.path.join(result_dir, 'alg_result.csv')
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                rew_vals = [float(r) for r in row_split[1:]]
                for i, r in enumerate(rew_vals):
                    if i not in reward_history.keys():
                        reward_history[i] = []
                        reported_rewards[i] = []
                    reward_history[i].append(r)
                
                    reported_reward = np.max(reward_history[i])
                    reported_rewards[i].append(reported_reward)

            row_index += 1

    # Use numpy data structs
    for key in reported_rewards.keys():
        reported_rewards[key] = np.array(reported_rewards[key])

    return np.array(experience), reported_rewards

def main():
    # Constants
    algs = ['PPO', 'A2C', 'DQN', 'SAC']
    envs = ['CustomCartPole-v0', 'CustomLunarLander-v0', 'CustomReacher-v0']
    working_dir = 'debug\\{}\\{}\\RANDOM_FIXED\\'

    for i_alg in algs:
        for j_env in envs:
            # Create data directory path 
            data_dir = working_dir.format(j_env, i_alg)

            # Read each seed directory
            cumualtive_results = {}
            seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]
            for seed_dir in seed_dirs:
                _, rewards = read_random_result_file(seed_dir, False)

                # Identify max score
                max_reward = -1e20
                for hp_rewards in rewards.values():
                    if max(hp_rewards) > max_reward:
                        max_reward = max(hp_rewards)

                # Choose reward thresholds
                threshold_percents = [0.25, 0.5, 0.75, 0.9]
                reward_thresholds = np.multiply(max_reward, threshold_percents)

                # Write line for each hyperparameter config
                for hp_config_id, hp_rewards in rewards.items():
                    for i_thresh, threshold in enumerate(reward_thresholds):
                        exp_index = bisect(hp_rewards, threshold)
                        if exp_index < hp_rewards.shape[0]:
                            if hp_config_id not in cumualtive_results.keys():
                                cumualtive_results[hp_config_id] = np.zeros_like(threshold_percents, dtype=int)
                            cumualtive_results[hp_config_id][i_thresh] += 1

            # Write summary file
            output_directory = 'random_investigation\\{}\\{}\\'.format(j_env, i_alg)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_file = '{}\\random.csv'.format(output_directory)
            with open(output_file, mode='w') as csv_file:
                # Write header
                header_line = [str(x) for x in threshold_percents]
                header_line.insert(0, 'HP Config')
                header_line = ','.join(header_line)
                csv_file.write(header_line + '\n')

                # Write line for each hyperparameter config
                for hp_config_id, threshold_counts in cumualtive_results.items():
                    # Write experience for each threshold
                    experience_line = [str(x) for x in threshold_counts]
                    experience_line.insert(0, str(hp_config_id))
                    experience_line = ','.join(experience_line)
                    csv_file.write(experience_line + '\n')


if __name__ == '__main__':
    main()