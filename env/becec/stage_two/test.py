'''
Author: your name
Date: 2022-02-21 22:04:51
LastEditTime: 2022-03-17 10:57:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /TSP_DRL_PtrNet-master/test.py
'''
import torch
from torch.utils.data import TensorDataset, DataLoader
from time import time
from env.becec.stage_two.env import Env_tsp
from env.becec.stage_two.config import Config, load_pkl, pkl_parser
from env.becec.stage_two.search import sampling, active_search
import pickle


class Test(object):
    """
		将需要的数据设置为初始数据
	"""

    def __init__(self, cfg, env, get_env):
        self.cfg = cfg
        self.env = env
        self.get_env = get_env
        self.score = None
        self.u = None
        self.trace = None

    def search_tour(self):
        data = \
            self.env.get_task_nodes(self.cfg.seed, self.get_env)
        total_cost, total_utility, trace = \
            sampling(self.cfg, self.env, data)
        self.score = total_cost
        self.u = total_utility
        self.trace = trace
