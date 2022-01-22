import os
import numpy as np

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
    median_rewards = []
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

                # Calcualte median rewards
                median_rewards.append(np.median(rew_vals))

            row_index += 1

    # Use numpy data structs
    for key in reported_rewards.keys():
        reported_rewards[key] = np.array(reported_rewards[key])

    return np.array(experience), reported_rewards

def main():
    # Constants
    algs = ['PPO', 'A2C', 'DQN', 'SAC']
    envs = ['CustomCartPole-v0', 'CustomLunarLander-v0', 'CustomReacher-v0']
    working_dir = 'results\\{}\\{}\\RANDOM_FIXED\\'

    for i_alg in algs:
        for j_env in envs:
            # Create data directory path 
            data_dir = working_dir.format(j_env, i_alg)

            # Read each seed directory
            min_length = 100000000
            seed_data = {}
            seed_dirs = [x[0] for x in os.walk(data_dir) if x[0] != data_dir]
            for seed_dir in seed_dirs:
                str_seed = seed_dir[-1]
                seed_data[str_seed] = read_random_result_file(seed_dir, False)

                # Identify min experience between seeds
                k_exp = seed_data[str_seed][0]
                if len(k_exp)-1 < min_length:
                    min_length = len(k_exp)-1

            # Find max reward within minimum experience
            max_score = -1e20
            for k_exp, k_rewards in seed_data.values():
                # Identify max
                for l_rewards in k_rewards.values():
                    if l_rewards[min_length] > max_score:
                        max_score = l_rewards[min_length]

            # Calculate score
            total_trials = 0
            above_threshold = 0
            threshold = 0.90 * max_score
            for k_exp, k_rewards in seed_data.values():
                # Identify if threshold was met
                for l_rewards in k_rewards.values():
                    if l_rewards[min_length] >= threshold:
                        above_threshold += 1
                    total_trials += 1 

            # Print results
            percent = (float(above_threshold) / float(total_trials)) * 100
            print('{}-{}: {}/{} ({}%)'.format(j_env, i_alg, above_threshold, total_trials,  percent))


if __name__ == '__main__':
    main()