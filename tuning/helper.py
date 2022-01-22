from .hoof import HOOF
from .pbt import PBT
from .sehop import SEHOP
from .random_fixed import RandomFixed
from .random_flex import RandomFlex

def create_tuning_strategy(config, env, device, run_ideal):
    if config['tuning_strategy_name'] == 'SEHOP':
        return SEHOP(config, env, device, run_ideal)
    elif config['tuning_strategy_name'] == 'HOOF':
        return HOOF(config, env, device, run_ideal)
    elif config['tuning_strategy_name'] == 'PBT':
        return PBT(config, env, device)
    elif config['tuning_strategy_name'] == 'RANDOM_FIXED':
        return RandomFixed(config, env, device)
    elif config['tuning_strategy_name'] == 'RANDOM_FLEX':
        return RandomFlex(config, env, device)

def load_tuning_strategy(config, directory_path, device):
    if config['tuning_strategy_name'] == 'SEHOP':
        return SEHOP.load(directory_path, device)
    elif config['tuning_strategy_name'] == 'HOOF':
        return HOOF.load(directory_path, device)
    elif config['tuning_strategy_name'] == 'PBT':
        return PBT.load(directory_path, device)
    elif config['tuning_strategy_name'] == 'RANDOM_FIXED':
        return RandomFixed.load(directory_path, device)
    elif config['tuning_strategy_name'] == 'RANDOM_FLEX':
        return RandomFlex.load(directory_path, device)