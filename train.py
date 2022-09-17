import argparse
from models.engine import load_engine
from utils.utils import read_config
from env.becec_wrapper import BECEC
from env.becec.stage_two.config import Config

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default='./config.yml', help="Path to the config file.")


if __name__ == "__main__":
    # args = vars(parser.parse_args())
    # config = read_config(args['config'])
    config = read_config('config_d4pg.yml')
    # if config['penalty_mode'] == 3 and config['is_null_penalty'] == False:
    #     # There is no negative penalty at all, and the lower bound on D4PG reward should be reset to 0
    #     config['v_min'] = 0

    BECEC.set_config(config)
    engine = load_engine(config)
    engine.train()