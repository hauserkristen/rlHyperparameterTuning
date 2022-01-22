import os
import argparse
import parse
from shutil import copyfile

def read_result_file(result_dir):
    result_file = os.path.join(result_dir, 'alg_sel.csv')

    experience = []
    reward = []
    with open(result_file, mode='r') as csv_file:
        row_index = 0
        for row in csv_file:
            row_split = row.split(',')     
            # Skip header      
            if row_index != 0:
                exp_val = float(row_split[2])
                rew_val = float(row_split[3])
                experience.append(exp_val)
                reward.append(rew_val)
            row_index += 1

    return experience, reward

def main():
    # Define main directory
    data_dir = 'results\\CustomLunarLander-v0\\DQN'
    destination_dir = os.path.join(data_dir, 'RANDOM_FIXED', '2')

    # Identify directories
    FILE_FRMT = 'HAND_TUNED_{alg_index:d}'
    config_dirs = {}
    for sub_dir in os.listdir(data_dir):
        parsed_dir = parse.parse(FILE_FRMT, sub_dir)
        if parsed_dir is not None:
            config_dirs[parsed_dir['alg_index']] = os.path.join(data_dir, sub_dir, '2')

    # Read each hand tuned file
    results = {}
    min_experience = None
    for alg_index, data_dir_index in config_dirs.items():
        # Read result file
        experience, reward = read_result_file(data_dir_index)

        # Store in history
        results[alg_index] = (experience, reward)

        # Record minimum experience from all results
        if min_experience is None:
            min_experience = len(experience)
        elif len(experience) < min_experience:
            min_experience = len(experience)

        # Copy initial alg select file to random
        if alg_index == 0:
            source_file = os.path.join(data_dir_index, 'alg_sel.csv')
            destination_file = os.path.join(destination_dir, 'alg_sel.csv')
            copyfile(source_file, destination_file)

    # Write single result file
    result_file = os.path.join(destination_dir, 'alg_result.csv')
    with open(result_file, mode='w') as csv_file:
        for i_experience in range(min_experience):
            # Initialize array for experience step
            experience_result = [0.0 for x in range(15)]

            # Gather info from each result
            for alg_index, (experience, reward) in results.items():
                experience_result[alg_index] = reward[i_experience]

            # Write header
            if i_experience == 0:
                header = ['Episode Number']
                for i_alg in range(15):
                    header.append('Reward {}'.format(i_alg))
                line = ','.join(header)
                csv_file.write(line + '\n')
            
            # Write line to file
            experience_result.insert(0, i_experience)
            experience_result = [str(x) for x in experience_result]
            line = ','.join(experience_result)
            csv_file.write(line + '\n')

if __name__ == '__main__':
    main()