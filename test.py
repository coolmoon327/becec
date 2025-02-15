import argparse
from models.engine import load_engine
from utils.utils import read_config
from env.becec_wrapper import BECEC
from env.becec.stage_two.config import Config
from env.becec.Test_Scheduler import Scheduler as Test_Scheduler

parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default='./config.yml', help="Path to the config file.")


if __name__ == "__main__":
    # args = vars(parser.parse_args())
    # config = read_config(args['config'])
    config = read_config('config_d4pg.yml')
    BECEC.set_config(config)
    if config['test_mode'] == 11:
        engine = load_engine(config)
        engine.test()
    else:
        test_scheduler = Test_Scheduler(config, config['test_mode'])
        test_scheduler.run()