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
            self.env.BS = i

            while len(tasks):
                if self.test.get_env.config['stage2_alg_choice'] == 0:
                    self.test.search_tour()  # 贪心
                elif self.test.get_env.config['stage2_alg_choice'] == 1:
                    self.test.active_search()  # dp
                score = self.test.score[0]
                u = self.test.u[0]
                trace = self.test.trace
                '''
                    score = -(u - c), u = sum(-alpha*dt), c = sum(p*w)
                '''

                if score == 5000.:
                    # print(f"target bs {i} has no more capacity!")
                    if penalty_mode == 0:
                        self.penalty += penalty * len(tasks)
                        self._env.clear_tasks_at_BS(i)

                        self.log_thrown_tasks_num += len(tasks)
                        break
                    elif penalty_mode == 1:
                        task_size = 0.
                        for t in range(len(tasks)):
                            # print(i, t, len(self._env.get_BS_tasks_external(BS_ID=i)))
                            task = tasks[t]
                            task_size += task.cpu_requirement()
                        bs_remain = 0.
                        for t in range(self._env.config["delta_t"]):
                            c = self._env.C(i, t)
                            bs_remain += c
                        throw_num = np.ceil(
                            (task_size - bs_remain) / (task_size / len(tasks)))
                        self.penalty += penalty * throw_num
                        self._env.clear_tasks_at_BS(i)

                        self.log_thrown_tasks_num += throw_num
                        break
                    elif penalty_mode == 2:
                        self.penalty += penalty
                        self.log_thrown_tasks_num += 1
                        tasks.pop(-1)
                    elif penalty_mode == 3 or penalty_mode == 4:
                        # 不对失误进行惩罚, 找出并删除最差的任务
                        self.log_thrown_tasks_num += 1
                        min_r = 1e6
                        target_task = -1
                        for t in range(len(tasks)):
                            uu, cc = cal_uc(i, tasks[t], trace[0][t])
                            r = uu - cc
                            if min_r > r:
                                min_r = r
                                target_task = t
                        tasks.pop(target_task)

                        # delta_t = self._env.config['delta_t']
                    # left_source = 0
                    # for t in range(delta_t):
                    #     left_source += self._env.C(i, t)
                    # a = left_source
                    # b = 0.
                    # for t in range(len(tasks)):
                    #     task = tasks[t]
                    #     b += task.cpu_requirement()
                    # # print(f"{a} < {b}")

                else:
                    c = u + score
                    self.cost += c
                    self.u += u
                    if c == 0:
                        print(f"Warning: cost is 0! c={c} u={u} score={score}")

                    for t in range(len(tasks)):
                        # print(i, t, len(self._env.get_BS_tasks_external(BS_ID=i)))
                        task = tasks[0]  # 每次 allocate 会删除任务，所以用 0
                        alloc_list = trace[0][t]
                        if sum(alloc_list) != task.cpu_requirement():
                            print("trace 和 task 的大小不匹配！")

                        self.u += task.u_0

                        uu, cc = cal_uc(i, task, alloc_list)
                        r = uu - cc
                        if r < 0:
                            tasks.remove(task)
                            # 还原 u 和 c
                            self.u += -uu
                            self.cost += -cc
                            if penalty_mode != 3:
                                # 将负的 reward 作为 penalty 进行训练, 方便在 pure reward 中查看手动剔除负 reward 分配的效果
                                # 如果是 3 模式, 则完全不考虑负 reward, 包括训练
                                self.penalty += r
                            # 完全不分配 u-c<0 的任务, 因为在测试中不会执行这类任务, 因此训练时不用考虑它们对环境的影响, 只用作为惩罚使用
                            continue

                        # delta_t = self._env.config['delta_t']
                        # left_source = 0
                        # for t in range(delta_t):
                        #     left_source += self._env.C(i, t)
                        # a = left_source

                        # allocate 会删除队列中的 task
                        self._env.allocate_task_at_BS(task=task, BS_ID=i,
                                                      alloc_list=alloc_list)

                        # left_source = 0
                        # for t in range(delta_t):
                        #     left_source += self._env.C(i, t)
                        # b = left_source
                        # # print(f"{a} - {b} = {task.cpu_requirement()}")

                    break

                # 只有在 penalty_mode >= 2 中，且无法完成调度的情况下，才会进行循环，直到剩下能够分配的任务序列
            else:
                pass

            if len(self._env.get_BS_tasks_external(BS_ID=i)):
                print(f"External tasks didn't be handled in BS {i}.")
                self._env.clear_tasks_at_BS(i)

        cost = self.cost
        u = self.u
        penalty = self.penalty
        return cost, u, penalty

    def get_thrown_tasks_num(self):
        return self.log_thrown_tasks_num
