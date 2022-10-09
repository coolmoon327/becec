'''
Author: your name
Date: 2022-02-21 22:04:51
LastEditTime: 2022-03-17 10:57:41
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /TSP_DRL_PtrNet-master/test.py
'''
import copy
import os
from datetime import datetime

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from time import time

from tqdm import tqdm

from env.becec.stage_two.actor import PtrNet1
from env.becec.stage_two.env import Env_tsp
from env.becec.stage_two.config import Config, load_pkl, pkl_parser
from env.becec.stage_two.search import sampling, active_search, dp
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

        self.train_steps = 0
        act_model = PtrNet1(cfg, get_env.config)
        self.act_model = act_model

    def search_tour(self):
        data = \
            self.env.get_task_nodes(self.cfg.seed, self.get_env)
        total_cost, total_utility, trace = \
            sampling(self.cfg, self.env, data)
        self.score = total_cost
        self.u = total_utility
        # trace[tours] (batch, task_seq) => (task_seq)
        trace['tours'] = trace['tours'].reshape(-1)
        # 统计未完成的任务数量
        trace['error_num'] = np.count_nonzero(trace['tours'] == -1)
        # trace['tours'] 中的 -1 全部删除掉
        trace['tours'] = np.delete(trace['tours'],
                                   np.where(trace['tours'] == -1))
        # trace 的顺序和 tours 改成一致的 trace (batch, task_seq, slots)
        trace['trace'] = trace['trace'][:, trace['tours'], :]
        # trace 的格式改成 (task_seq, slots)
        if len(trace['tours']) == 0:  # 没有能完成的任务
            trace['trace'] = np.array([])
        else:
            trace['trace'] = trace['trace'].reshape(len(trace['tours']), -1)

        self.trace = trace

    def dp_search(self):
        data = \
            self.env.get_task_nodes(self.cfg.seed, self.get_env)
        total_cost, total_utility, trace = \
            dp(self.cfg, self.env, data)
        self.score = total_cost
        self.u = total_utility
        # trace[tours] (batch, task_seq) => (task_seq)
        trace['tours'] = trace['tours'].reshape(-1)
        # 统计未完成的任务数量
        trace['error_num'] = np.count_nonzero(trace['tours'] == -1)
        # trace['tours'] 中的 -1 全部删除掉
        trace['tours'] = np.delete(trace['tours'],
                                   np.where(trace['tours'] == -1))
        # trace 的顺序和 tours 改成一致的 trace (batch, task_seq, slots)
        trace['trace'] = trace['trace'][:, trace['tours'], :]
        # trace 的格式改成 (task_seq, slots)
        if len(trace['tours']) == 0:  # 没有能完成的任务
            trace['trace'] = np.array([])
        else:
            trace['trace'] = trace['trace'].reshape(len(trace['tours']), -1)

        self.trace = trace

    def network_train(self):
        """
        可以获取环境信息和任务的输入信息, 可以在这个位置训练整个网络
        :return:
        """
        pass

    def dp(self):
        pass

    def active_search(self):
        '''
        active search updates model parameters even during inference on a single input
        test input:(city_t,xy)
        '''
        # 获取 task 和 env 的训练信息, tasks 中包含了任务信息和环境信息
        tasks = self.env.get_task_nodes(self.cfg.seed, self.get_env)

        # 获得随机的任务执行顺序
        batch, _, tasks_slots_info = tasks.size()
        random_tours = self.env.stack_random_tours(batch)

        # 为随机的任务执行顺序打分, 得到一个基准, 同时需要传入 tasks 和 envs
        baseline = self.env.seq_score(tasks, random_tours)
        self.score = baseline[0].cpu().numpy()
        self.u = baseline[1]
        # 这里的 trace 需要重新将任务的执行顺序 变为 0, 1, 2, 3, 4 ...
        # 维度还是需要是 (batch, task_size, slots)
        # 需要重新调整一下顺序
        self.trace = baseline[2][:, random_tours[0][:], :]
        self.trace = np.reshape(self.trace, (batch, random_tours.shape[1],
                                             (tasks_slots_info - 2) // 2))
        """
        # 记录成本
        cost = baseline[0]

        # 设置优化器
        act_model = self.act_model
        act_optim = optim.Adam(act_model.parameters(),
                               lr=self.get_env.config[
                                   'actor_learning_rate'])

        # 使用模型
        act_model = act_model.to(self.get_env.config['device'])

        # 执行模型
        task_data, env_data = torch.split(tasks, dim=2,
                                          split_size_or_sections=[
                                              tasks_slots_info - self.env.slots * 2,
                                              self.env.slots * 2])
        env_data = env_data[:, 0, :]
        pred_shuffle_tours, neg_log = act_model((task_data, env_data),
                                                self.get_env.config[
                                                    'device'])

        # 对任务执行顺序进行评价
        l_batch = self.env.seq_score(tasks, pred_shuffle_tours)

        # 比基本的方法要好多少
        adv = l_batch[0] - baseline[0]
        act_optim.zero_grad()
        act_loss = torch.mean(adv * neg_log)
        act_loss.backward()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1.,
                                 norm_type=2)
        act_optim.step()
        baseline = baseline[0] * self.cfg.alpha + (1 - self.cfg.alpha) * \
                   torch.mean(
            l_batch[0], dim=0)
        # print("act_loss is %f" % (act_loss.data))
        # 用 tensorboard 替换
        # print(
        #     'step:%d/%d, actic loss:%1.3f' % (
        #         i, cfg.steps, act_loss.data))
        """
