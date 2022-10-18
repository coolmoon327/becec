'''
Author: your name
Date: 2022-03-04 14:29:08
LastEditTime: 2022-03-17 10:58:36
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /own_work/stage_two/Stage_Two_Pointer.py
'''
import numpy as np
import os
import sys
import copy

from env.becec.stage_two.config import Config, load_pkl, pkl_parser, argparser, \
    dump_pkl
from env.becec.stage_two.env import Env_tsp
from env.becec.stage_two.test import Test

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from env.becec.Environment import Environment

noise_scale = 0.3
batch_size = 128
updates_per_step = 5


class Stage_Two_Pointer:
    def __init__(self, env: Environment):
        '''
            self._env 是全局 env
            self.env 是二阶段 env
        :param env:
        '''
        self._env = env
        self.cfg = argparser(env)
        self.env = Env_tsp(self.cfg)
        self.test = Test(self.cfg, self.env, self._env)
        self.cost = 0.
        self.u = 0.
        self.penalty = 0.

        self.log_thrown_tasks_num = 0

        self.test_mode = False

    def reset(self):
        self.cost = 0.
        self.u = 0.
        self.penalty = 0.
        self.log_thrown_tasks_num = 0

    def execute(self):
        """
        执行预测
        需要在外部记录 state 和 action
        :return:        numpy 格式的 s a
        """
        '''
            cfg 放在外面,只做一次初始化就可以了
        '''
        self.cost = 0.
        self.u = 0.
        self.penalty = 0.

        self.log_thrown_tasks_num = 0

        penalty = self._env.config['penalty']
        penalty_mode = self._env.config['penalty_mode']

        def cal_uc(i, task, alloc_list):
            env = self._env
            end_t = 0
            c = 0.
            for t in range(env.config['delta_t']):
                c += env.p(i, t) * alloc_list[t]
                if alloc_list[t] > 0:
                    end_t = t
            u = task.utility(env.timer + end_t)
            return u, c

        for i in range(self._env.config['M']):
            '''
                self._env.get_BS_tasks_external(BS_ID=i)
                任务获取的过程并不会清空任务列表
            '''

            tasks = self._env.get_BS_tasks_external(BS_ID=i)
            tasks = copy.copy(tasks)    # 二阶段进行了更改, 因此不再允许 tasks 队列随着分配被删除, 需要拿到一个独立于 _env 的队列
            tasks_num = len(tasks)
            if tasks_num == 0:
                continue

            # self.test.trainRl() # 训练第二阶段
            self.env.BS = i
            if self.test.get_env.config['stage2_alg_choice'] == 0:
                self.test.search_tour()  # 贪心
            elif self.test.get_env.config['stage2_alg_choice'] == 1:
                self.test.dp_search()  # dp
            # self.test.search_tour()
            # print(f"greedy: {self.test.score[0]}")
            # self.test.dp_search()
            # print(f"dp    : {self.test.score[0]}")
            # print()

            score = self.test.score[0]
            u = self.test.u[0]
            trace = self.test.trace['trace']
            tours = self.test.trace['tours']
            error_num = self.test.trace['error_num']

            if error_num > 0:
                # 0 和 1 都不执行整个调度
                if penalty_mode == 0:
                    self.penalty += penalty * tasks_num
                    self._env.clear_tasks_at_BS(i)
                    self.log_thrown_tasks_num += tasks_num
                elif penalty_mode == 1:
                    self.penalty += penalty * error_num
                    self._env.clear_tasks_at_BS(i)
                    self.log_thrown_tasks_num += error_num
                # 2 3 4 会尽可能执行调度
                elif penalty_mode == 2:
                    self.penalty += penalty * error_num
                    self.log_thrown_tasks_num += error_num
                elif penalty_mode == 3 or penalty_mode == 4:
                    # 不对失误进行惩罚
                    self.log_thrown_tasks_num += error_num

            c = u + score
            self.cost += c
            self.u += u
            # if c == 0:
            #     print(f"Warning: cost is 0! c={c} u={u} score={score}")

            for index in range(len(tours)):
                tid = tours[index]
                task = tasks[tid]
                alloc_list = trace[index]
                if sum(alloc_list) != task.cpu_requirement():
                    print("ERROR: trace 和 task 的大小不匹配！")

                self.u += task.u_0

                if ((penalty_mode == 3 or penalty_mode == 4) and not self.test_mode) or self._env.config['force_ignore_bad_tasks']:
                    uu, cc = cal_uc(i, task, alloc_list)
                    r = uu - cc
                    if r < 0:
                        # 还原 u 和 c
                        self.u += -uu
                        self.cost += -cc
                        # 如果是 3 模式, 则完全不考虑负 reward, 包括训练
                        if penalty_mode == 4:
                            # 将负的 reward 作为 penalty 进行训练, 方便在 pure reward 中查看手动剔除负 reward 分配的效果
                            self.penalty += r
                        self.log_thrown_tasks_num += 1
                        continue # 完全不分配 u-c<0 的任务, 因为在测试中不会执行这类任务, 因此训练时不用考虑它们对环境的影响, 只用作为惩罚使用

                # allocate 会删除队列中的 task, 因此需要在获取 tasks 时进行 deepcopy
                self._env.allocate_task_at_BS(task=task, BS_ID=i, alloc_list=alloc_list)

            if len(self._env.get_BS_tasks_external(BS_ID=i)):
                # print(f"External tasks didn't be handled in BS {i}.")  # 有这个报错也正常, 说明 r<0 被丢弃了, 或者存在错误分配
                self._env.clear_tasks_at_BS(i)

        return self.cost, self.u, self.penalty

    def get_thrown_tasks_num(self):
        return self.log_thrown_tasks_num
