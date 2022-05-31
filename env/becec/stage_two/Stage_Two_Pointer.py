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

from env.becec.stage_two.config import Config, load_pkl, pkl_parser, argparser, dump_pkl
from env.becec.stage_two.env import Env_tsp
from env.becec.stage_two.test import Test
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from env.becec.Environment import Environment
noise_scale = 0.3
batch_size = 128
updates_per_step = 5

penalty_mode = 1

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
        self.cost = 0
        self.u = 0

    def execute(self):
        """
        执行预测
        需要在外部记录 state 和 action
        :return:        numpy 格式的 s a
        """
        '''
            cfg 放在外面,只做一次初始化就可以了
        '''
        for i in range(self._env.config['M']):
            self.env.BS = i
            '''
                self._env.get_BS_tasks_external(BS_ID=i)
                任务获取的过程并不会清空任务列表
            '''

            tasks = self._env.get_BS_tasks_external(BS_ID=i)


            num = len(tasks)
            if num:
                self.test.search_tour()
                score = self.test.score[0]
                u = self.test.u[0]
                trace = self.test.trace
                '''
                    score = -(u - c), u = sum(-alpha*dt), c = sum(p*w)
                '''

                if score == 5000.:
                    # print(f"target bs {i} has no more capacity!")
                    if penalty_mode == 0:
                        self.u += -1000. * num
                    elif penalty_mode == 1:
                        task_size = 0.
                        for t in range(num):
                            # print(i, t, len(self._env.get_BS_tasks_external(BS_ID=i)))
                            task = tasks[t]
                            task_size += task.cpu_requirement()
                        bs_remain = 0.
                        for t in range(self._env.config["delta_t"]):
                            c = self._env.C(i, t)
                            bs_remain += c
                        throw_num = np.ceil((task_size-bs_remain) / (task_size/num))
                        self.u += -1000. * throw_num
                    # 清空队列
                    self._env.clear_tasks_at_BS(i)
                else:
                    c = u + score
                    self.cost += c
                    self.u += u

                    for t in range(num):
                        # print(i, t, len(self._env.get_BS_tasks_external(BS_ID=i)))
                        task = tasks[0]  # 每次 allocate 会删除任务，所以用 0
                        alloc_list = trace[0][t]
                        if sum(alloc_list) != task.cpu_requirement():
                            print("trace 和 task 的大小不匹配！")

                        self.u += task.u_0

                        # allocate 会删除队列中的 task
                        self._env.allocate_task_at_BS(task=task, BS_ID=i, alloc_list=alloc_list)
            else:
                pass
        cost = self.cost
        u = self.u
        self.cost = self.u = 0.
        return cost, u