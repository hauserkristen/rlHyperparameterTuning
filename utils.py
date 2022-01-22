import os
import numpy as np
import bisect

# Max reward for envs
MAX_REWARDS = {
    'CustomCartPole-v0': 200
}

def read_result_file(result_dir: str, use_rolling_avg: bool):
    result_file = os.path.join(result_dir, 'alg_sel.csv')

    experience = []
    reported_rewards = []
    reward_history = []
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[2])
                rew_val = float(row_split[3])
                experience.append(exp_val)
                reward_history.append(rew_val)
                
                if use_rolling_avg:
                    reported_reward = np.mean(reward_history[-100:])
                else:
                    reported_reward = np.max(reward_history)
                reported_rewards.append(reported_reward)

            row_index += 1

    return np.array(experience), np.array(reported_rewards)

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
                
                    if use_rolling_avg:
                        reported_reward = np.mean(reward_history[i][-100:])
                    else:
                        reported_reward = np.max(reward_history[i])
                    reported_rewards[i].append(reported_reward)

            row_index += 1

    # Calculate median
    median_rewards = []
    history_length = len(list(reward_history.values())[0])
    for i in range(history_length):
        i_history = [r[i] for r in reward_history.values()]
        median_rewards.append(np.mean(i_history))
        
    # Identify single setting most similar to median sequence
    min_simularity = None
    min_index = None
    for i, r in reward_history.items():
        simularity = calculate_simularity(median_rewards, r)

        if min_simularity is None or simularity < min_simularity:
            min_simularity = simularity
            min_index = i


    return np.array(experience), np.array(reported_rewards[min_index])

def read_ideal_result_file(result_dir: str, use_rolling_avg: bool):
    # Record reward per index
    reported_rewards = []
    reward_history = []
    alg_sel = []
    result_file = os.path.join(result_dir, 'ideal_sel.csv')
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                sel_val = int(row_split[1])
                rew_val = float(row_split[2])
                
                alg_sel.append(sel_val)
                reward_history.append(rew_val)
                
                if use_rolling_avg:
                    reported_reward = np.mean(reward_history[-100:])
                else:
                    reported_reward = np.max(reward_history)
                reported_rewards.append(reported_reward)
            row_index += 1

    return np.array(reported_rewards), np.array(alg_sel)

def read_alg_sel(result_dir: str):
    result_file = os.path.join(result_dir, 'alg_sel.csv')

    alg_sel = []
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                sel_val = int(row_split[1])
                alg_sel.append(sel_val)

            row_index += 1

    return np.array(alg_sel)

def extend_experience(index_data: dict, env: str, max_experience: int):
    if env in MAX_REWARDS.keys():
        for index in index_data.keys():
            e, r = index_data[index]
            if r[-1] == MAX_REWARDS[env]:
                e = np.append(e, [max_experience])
                r = np.append(r, [MAX_REWARDS[env]])
                index_data[index] = e, r

    return index_data

def average_across_trials(index_data: dict, use_median: bool, env: str, max_experience: int):
    # Identify miniumum of maximum for upper bound of experience
    index_minimax_e = min([max(e) for e, r in index_data.values()])

    min_reward = []
    max_reward = []
    avg_reward = []
    avg_experience = []
    RECORD_INTERVAL = 100
    for x_val in np.arange(RECORD_INTERVAL, index_minimax_e+RECORD_INTERVAL, RECORD_INTERVAL):
        index_vals = []
        for e, r in index_data.values():
            exp_index = bisect.bisect(e, x_val)
            if exp_index < r.shape[0]:
                r_val = r[exp_index]
            else:
                r_val = r[-1]
            index_vals.append(r_val)

        # Calculate mid value
        if use_median:
            avg_reward.append(np.median(index_vals))
        else:
            avg_reward.append(np.mean(index_vals))

        # Calculate min and max
        min_reward.append(np.percentile(index_vals, 5))
        max_reward.append(np.percentile(index_vals, 95))

        avg_experience.append(x_val)

    # If using median, extend again
    if use_median and env in MAX_REWARDS.keys():
        if avg_reward[-1] == MAX_REWARDS[env]:
            avg_reward.append(MAX_REWARDS[env])
            avg_experience.append(max_experience)
            min_reward.append(min_reward[-1])
            max_reward.append(max_reward[-1])


    return np.array(avg_experience), np.array(avg_reward), np.array(min_reward), np.array(max_reward)

def calculate_simularity(arr_one: list, arr_two: list):
    return np.square(np.subtract(arr_one, arr_two)).mean()