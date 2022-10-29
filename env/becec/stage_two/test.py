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
from torch.utils.data import TensorDataset, DataLoader, Dataset
from time import time

from tqdm import tqdm

from env.becec.stage_two.Greedy import Greedy
from env.becec.stage_two.actor import PtrNet1
from env.becec.stage_two.env import Env_tsp
from env.becec.stage_two.config import Config, load_pkl, pkl_parser
from env.becec.stage_two.search import sampling, active_search, dp, \
    compare_pointer
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
        act_model.load_state_dict(
            torch.load(
                "env/becec/stage_two/Pt/train10_0423_14_28_step9999_act.pt"))
        self.act_model = act_model
        self.act_model.to(torch.device("cuda:0" if torch.cuda.is_available()
                                       else "cpu"))

    def search_tour(self):
        data = \
            self.env.get_task_nodes(self.cfg.seed, self.get_env)
        # total_cost, total_utility, trace = \
        #     compare_pointer(self.cfg, self.env, data, self.act_model)
        # print(f"pointer totoal: {total_utility[0] - total_cost[0]}")
        total_cost, total_utility, trace = sampling(self.cfg, self.env, data)
        # print(f"对比算法  totoal: {total_utility[0] - total_cost[0]}")
        # print()
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
        # # trace 的顺序和 tours 改成一致的 trace (batch, task_seq, slots)
        # trace['trace'] = trace['trace'][:, trace['tours'], :]
        # trace 的格式改成 (task_seq, slots)
        if len(trace['tours']) == 0:  # 没有能完成的任务
            trace['trace'] = np.array([])
            self.search_tour()
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
        # 这两个位置, 对任务执行顺序的评价可以放宽一点 Σuct就可以了 
        # 那么甚至下面的adv函数都不需要使用了
        # act_loss也非常容易写出来
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

    def trainRl(self):
        '''训练整个二阶段的RL'''
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.getcwd() + '/env/becec/stage_two/Pt/run')
        batch = 32
        # 获取env和task的信息, 这里自己来生成, task固定长度, env的shape也需要关注
        dataset = BSDataset(self.cfg)
        train_loader = DataLoader(dataset=dataset, batch_size=batch,
                                  shuffle=True, num_workers=2)

        # 设置优化器
        act_model = self.act_model
        act_optim = optim.Adam(act_model.parameters(),
                               lr=self.get_env.config[
                                   'actor_learning_rate'])
        criterion = torch.nn.L1Loss()

        # 使用模型
        act_model = act_model.to(self.get_env.config['device'])

        # 在 RL 训练中还是可以使用 baseline
        # 获得随机的任务执行顺序
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        min_epoch = 0
        min_loss = float('inf')
        for epoch in range(dataset.epoch):
            random_tours = dataset.random_tours()
            # baseline = 0
            for batch_idx, (task_data, env_data) in enumerate(train_loader):
                # if batch_idx == 0:
                #     baseline = dataset.uct(task_data, random_tours)
                #     continue

                tours, neg_log = act_model((task_data, env_data),
                                           device=device)

                # 这两个位置, 对任务执行顺序的评价可以放宽一点 Σuct就可以了
                act_optim.zero_grad()
                # act_loss = Σuct
                # 30000 基准
                act_loss = criterion(dataset.uct(task_data, tours),
                                     torch.tensor(0))
                act_loss = act_loss.requires_grad_(True)
                # act_loss = np.abs(dataset.uct(task_data, tours) - baseline)
                act_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1.,
                                         norm_type=2)
                # 步进
                act_optim.step()

                # 保存最佳的模型
                if epoch >= min_epoch and act_loss < min_loss:
                    torch.save(act_model.state_dict(),
                               os.getcwd() + '/env/becec/stage_two/Pt/%d.pth' % (
                                   epoch))
                    min_epoch = epoch
                    min_loss = act_loss

                # 打印和保存最佳模型的地方
                if (batch_idx + 1) % 10 == 0:
                    writer.add_scalar('Loss/train', act_loss.item(),
                                      epoch * (dataset.n // dataset.batch) +
                                      batch_idx)
                    # print(epoch, batch_idx, act_loss.item())


class BSDataset(Dataset):
    def __init__(self, config):
        # 10240000
        n = 1024000
        task_num = config.city_t
        slot_num = config.slots
        # w [5, 20]
        w = np.random.randint(5, 21, size=(n, task_num, 1))  # [5,
        # 21]
        # alpha [10, 90]
        alpha = np.random.randint(10000, 90001, size=(n, task_num, 1)) / 1000.

        # c [15, 25]
        c = np.random.randint(15000, 25000, size=(n, slot_num, 1)) / 1000.
        # p [0.5, 1] / (GHz)
        p = np.random.randint(500, 1000, size=(n, slot_num, 1)) / 1e6

        self.task_data = torch.from_numpy(np.concatenate((w, alpha), axis=2))
        self.env_data = torch.from_numpy(np.concatenate((c, p), axis=2))
        self.task_data = self.task_data.float()
        self.env_data = self.env_data.float()
        self.n = n
        self.task_num = task_num
        self.slot_num = slot_num
        self.batch = 32
        self.epoch = 1000

    def __getitem__(self, index):
        """通过索引得到对象"""
        return self.task_data[index], self.env_data[index]

    def __len__(self):
        """得到对象的长度"""
        return self.n

    def random_tours(self):
        '''
        tour:(city_t)
        return tours:(batch,city_t)
        '''
        list = [self.get_random_tour() for i in range(self.batch)]
        tours = torch.stack(list, dim=0)
        return tours

    def get_random_tour(self):
        tour = []
        while set(tour) != set(range(self.task_num)):
            task = np.random.randint(self.task_num)
            if task not in tour:
                tour.append(task)
        tour = torch.from_numpy(np.array(tour))
        return tour

    def uct(self, task_info, tours):
        """一批任务的loss计算Σuct
        :param task_info (batch, 10, 2)
        :param tours (batch, 10)
        """

        u_c_t = np.zeros(self.batch, )
        t_slots = np.arange(0, tours.shape[1])
        for b in range(self.batch):
            u_c_t[b] = np.dot(task_info[b][tours[b]][:, 1] * task_info[b][tours[
                b]][:, 0], t_slots)
        return torch.from_numpy(u_c_t).float().mean()

    def greedy(self, task_data, env_data, tours):
        """
        计算在greedy情况下的 u-c的值
        :param task_data (batch, 10, 2)
        :param env_data (batch, 10, 2)
        :param tours (batch, 10)
        """
        Greedy()
