from curses.ascii import BS
import numpy as np
from typing import List
import pickle
import os

from .BaseStation import BaseStation
from .Task import Task


class Environment:
    """
        A class used to describe the dynamic environment

        Attributes:
            timer:                                  record present time slot number
            frame_timer:                            record the position of the current moment in a frame
            task_set:                               new tasks in this frame
            task_batch_num:                         当前正在处理第几批任务

        a task's attr is {'t': int, 's': int,'k': float, 'b': float}
        where t is the arrival time slot number, s is the data length, value function = k * tau + b
    """

    def __init__(self, config):
        self.config = config
        self.BS = [BaseStation(BS_ID=j, config=config) for j in range(config['M'])]
        self.timer = 0
        self.frame_timer = 0
        self.task_set = []
        self.task_batch_num = 0

    def reset(self):
        self.timer = 0
        self.frame_timer = 0
        self.task_set = []
        self.task_batch_num = 0
        for bs in self.BS:
            bs.reset()

    def _get_frame_end_slot(self):
        """
        获取当前 timer 所在 frame 中的最后一个 slot 的编号
        :return:    the slot number
        """
        frame = self.config['frame']
        return frame - 1 - self.frame_timer + self.timer

    def is_resource_enough_in_BS(self, task: Task, BS_ID: int):
        """该方法用于第一阶段调度
        判断当前 task 是否能够调度给第 BS_ID 个基站
        :param task:
        :param BS_ID:
        :return:
        """
        if not 0 <= BS_ID < self.config['M']:
            print(f"Parameter BS={BS_ID} is out of range! Return false.")
            return False
        return self.BS[BS_ID].is_resource_enough(task=task)

    def is_resource_enough_in_BS_at_slots(self, task: Task, BS_ID: int, slots_begin: int, slots_end: int):
        """该方法用于第二阶段分配
        判断当前 task 是否能够分配在第 BS_ID 个基站的 [slots_begin, slots_end] 时隙中
        :param task:
        :param BS_ID:
        :param slots_begin:              the first slot number [0, delta_t - 1]
        :param slots_end:                the last slot number [0, delta_t - 1]
        :return:
        """
        if not 0 <= BS_ID < self.config['M']:
            print(f"Parameter BS={BS_ID} is out of range! Return false.")
            return False
        return self.BS[BS_ID].is_resource_enough_at_slots(task=task, slots_begin=slots_begin, slots_end=slots_end)

    def schedule_task_to_BS(self, task: Task, BS_ID: int):
        """该方法用于第一阶段调度
        将任务按照调度列表分配给某个基站
        :param task:
        :param BS_ID:                   the number of target helper BS
        :return:                        0 for success, -1 for failure
        """
        if not 0 <= BS_ID < self.config['M']:
            print(f"Parameter BS={BS_ID} is invalid!")
            return -1
        return self.BS[BS_ID].schedule_task(task=task)

    def allocate_task_at_BS(self, task: Task, BS_ID: int, alloc_list: List[int]):
        """该方法用于第二阶段分配
        将任务按照调度列表分配给某个基站
        :param task:
        :param BS_ID:                   the number of target helper BS
        :param alloc_list:              the allocation for task in this BS during the next 0 to delta_t - 1 slots. len = delta_t
        :return:                        0 for success, -1 for failure
        """
        if not 0 <= BS_ID < self.config['M']:
            print(f"Parameter BS={BS_ID} is out of range!")
            return -1
        if task not in self.get_BS_tasks_external(BS_ID=BS_ID):
            print(f"The parameter task passed in is not in the cache of the BS{BS_ID}")
            return -1
        return self.BS[BS_ID].allocate_task(task=task, alloc_list=alloc_list)

    def get_BS_tasks_external(self, BS_ID: int):
        """该方法用于第二阶段分配
        获取第 BS_ID 个基站中当前 timer 被调度的外部任务缓存列表
        :param BS_ID:
        :return:
        """
        if not 0 <= BS_ID < self.config['M']:
            print(f"Parameter BS={BS_ID} is out of range! Return false.")
            return []
        return self.BS[BS_ID].tasks_external

    def is_end_of_frame(self):
        """
            Determine if this timer is the end of a frame
        """
        return self._get_frame_end_slot() == self.timer

    def next(self):
        self.task_batch_num = 0
        self.timer += 1
        self.frame_timer = (self.frame_timer + 1) % self.config['frame']
        if self.frame_timer == 0:
            # 新 frame 开始清空任务队列
            self.task_set.clear()

        frame_end_slot = self._get_frame_end_slot()
        for bs in self.BS:
            tasks = bs.next()
            # 把 task 的到达时间重新设置为 frame 中最后一个 slot 的编号
            for task in tasks:
                task.arrival_time = frame_end_slot
            self.task_set += tasks

    def C(self, i: int, t: int):
        """
        C: M * delta_t. C(i, t) indicates available resources of BS i at slot t
        :param i:       BS number           [0, M-1]
        :param t:       offset slot number  [0, delta_t-1]
        :return:        available resources of BS i at slot t
        """
        # 错误则直接返回 0，表示无资源
        if not 0 <= i < self.config['M']:
            print(f"Parameter BS={i} is out of range!")
            return 0
        if not 0 <= t <= self.config['delta_t']:
            print(f"Parameter slot={t} is out of range! Return the list of cpu_remain in BS{i}.")
            return 0

        return self.BS[i].cpu_remain(t=t)

    def p(self, i: int, t: int):
        """
        p: M * delta_t. p(i, t) indicates unit price of BS i at slot t
        :param i:       BS number           [0, M-1]
        :param t:       offset slot number  [0, delta_t-1]
        :return:        unit price of BS i at slot t
        """
        # 错误或 cpu_remain 为 0 则直接返回 1，表示资源不出售
        if not 0 <= i < self.config['M']:
            print(f"Parameter BS={i} is out of range!")
            return 1.
        if not 0 <= t <= self.config['delta_t']:
            print(f"Parameter slot={t} is out of range! Return the list of cpu_remain in BS{i}.")
            return 1.

        return self.BS[i].unit_price(t=t)

    def is_local_file_exsisted(self):
        env_name = f"M{self.config['M']}T{self.config['T']}delta_t{self.config['delta_t']}"
        suffix = 1
        path = "env/becec/data/environment_{}_{}".format(env_name, suffix)
        return os.path.isfile(path)
    
    def loadEnv(self):
        env_name = f"M{self.config['M']}T{self.config['T']}delta_t{self.config['delta_t']}"
        suffix = 1
        path = "env/becec/data/environment_{}_{}".format(env_name, suffix)
        
        with open(path, 'rb') as f:
            self.BS = pickle.load(f)
            
        for bs in self.BS:
            bs.reset()  # 清空队列

    def saveEnv(self):
        # 主要保存 BS 数据
        if not os.path.exists('env/data/'):
            os.makedirs('env/data/')
        
        env_name = f"M{self.config['M']}T{self.config['T']}delta_t{self.config['delta_t']}"
        suffix = 1
        path = "env/becec/data/environment_{}_{}".format(env_name, suffix)
        
        with open(path, 'ab') as f:
            pickle.dump(self.BS, f)

    # def setSeed(self, seed: int):
    #     pass

    def next_task_batch(self):
        """
        在 scheduler 中将任务进行分批，每处理完一批task，执行一次该方法
        :return:        当前批是否已经不可行（已经没有对应的任务）
        """
        self.task_batch_num += 1

        return self.task_batch_num  * self.config['n_tasks'] + 1 > len(self.task_set)

    def clear_tasks_at_BS(self, BS_ID: int):
        self.BS[BS_ID].clear_tasks()

    def seed(self, seed):
        np.random.seed(seed)
        for bs in self.BS:
            bs.seed(seed+bs.uuid)   # 避免所有 BS 的 seed 完全一样，导致随机的内容一样
        # TODO 检查是否还有其他地方需要设置随机种子