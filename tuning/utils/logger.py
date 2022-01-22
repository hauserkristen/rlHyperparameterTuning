import os
import numpy as np

class Logger(object):
    def __init__(self, result_directory: str, run_ideal: bool, initialize_files: bool = True):
        self.run_ideal = run_ideal

        # Create directory if does not exist
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)

        # Create file names
        self.alg_definition_file = os.path.join(result_directory, 'alg_def.csv')
        self.alg_selection_file = os.path.join(result_directory, 'alg_sel.csv')
        self.ideal_selection_file = os.path.join(result_directory, 'ideal_sel.csv')
        
        # Initialize files
        if initialize_files:
            self._initialize_files()

    def _initialize_files(self):
        # Delete previous version
        if os.path.exists(self.alg_definition_file):
            os.remove(self.alg_definition_file)

        if os.path.exists(self.alg_selection_file):
            os.remove(self.alg_selection_file)

        if os.path.exists(self.ideal_selection_file):
            os.remove(self.ideal_selection_file)

    def reset(self, alg_configs):
        # Reinitialize files
        self._initialize_files()

        # Write algorithm definition file
        self.write_initial_files(alg_configs)

    def log_selection(self, episode_num: int, alg_index: int, experience: int, expected_reward: float, reward_variance: float):
        with open(self.alg_selection_file, mode='a') as f:
            row = [str(episode_num), str(alg_index), str(experience), str(expected_reward), str(reward_variance)]
            row_str = ','.join(row)
            f.write(row_str + '\n')

    def log_ideal_selection(self, episode_num: int, alg_index: int, expected_reward: float):
        with open(self.ideal_selection_file, mode='a') as f:
            row = [str(episode_num), str(alg_index), str(expected_reward)]
            row_str = ','.join(row)
            f.write(row_str + '\n')

    def write_initial_files(self, alg_configs):
        # Write header to selection file
        column_header = ['Episode Number', 'Algorithm ID', 'Experience', 'Expected Reward', 'Reward Variance']
        with open(self.alg_selection_file, mode='w') as f:
            # Write header
            header = ','.join(column_header)
            f.write(header + '\n')

        # Write algorithm configurations
        cfg_keys = list(alg_configs[0].keys())
        column_header = ['Algorithm ID'] + cfg_keys
        with open(self.alg_definition_file, mode='w') as f:
            # Write header
            header = ','.join(column_header)
            f.write(header + '\n')

            for i, a_cfg in enumerate(alg_configs):
                # Get parameter values
                row = [str(a_cfg[k]) for k in cfg_keys]
                row.insert(0, str(i))

                # Write row
                row_str = ','.join(row)
                f.write(row_str + '\n')


        # Write ideal selection if applicable
        if self.run_ideal:
            column_header = ['Episode Number', 'Ideal Algorithm ID', 'Ideal Reward']
            with open(self.ideal_selection_file, mode='w') as f:
                # Write header
                header = ','.join(column_header)
                f.write(header + '\n')
