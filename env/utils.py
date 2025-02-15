from .pendulum import PendulumWrapper
from .bipedal import BipedalWalker
from .env_wrapper import EnvWrapper
from .lunar_lander_continous import LunarLanderContinous
from .becec_wrapper import BECEC


def create_env_wrapper(config):
    env_name = config['env']
    if env_name == "becec" or env_name == "BECEC":
        return BECEC(config)
    elif env_name == "Pendulum-v0":
        return PendulumWrapper(config)
    elif env_name == "BipedalWalker-v2":
        return BipedalWalker(config)
    elif env_name == "LunarLanderContinuous-v2":
        return LunarLanderContinous(config)
    return EnvWrapper(env_name)