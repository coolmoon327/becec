import numpy as np
import torch
import copy


class Greedy(object):
    def __init__(self, inputs, tours):
        """
            :param inputs: (batch, city_t, 2 + slots * 2)
                workload, alpha, c, p
            :param tours: (batch, city_t), predicted orders
        """

        '''
            1.取出数据
            2.数据格式的转换
        '''
        '''
            最简单的改变办法是将 inputs 拆分成 task_data 和 env_data
            使用方法 np.split
            task_data 的信息固定是维的
            env_data 是后面的信息
            同时需要对 env_data 进行降解
        '''
        '''
            测试第二阶段时都不用 GPU
        '''
        data = copy.deepcopy(inputs)
        data = data.numpy()
        _, task_size = tours.size()
        task_data, env_data = np.split(data, axis=2, indices_or_sections=[
            task_size * 2])
        # 这里的 indices_or_sections 是不是应该要变为数据的实际长度
        env_data = env_data[:, 0, :]
        '''
            此时传入的 tours 已经是numpy类型了
            不用再转换为numpy类型
        '''
        tours = tours.numpy()  # 先将输入转为numpy类型
        c_slots, p_slots = np.split(env_data, 2, axis=1)
        workload, penalty_factor = np.split(task_data, 2, axis=2)
        workload = np.squeeze(workload, axis=-1)  # 降解最后一个维度
        penalty_factor = np.squeeze(penalty_factor, axis=-1)

        self.task_data = task_data
        self.env_data = env_data
        self.tours = tours
        self._c_slots = c_slots
        self._p_slots = p_slots
        self._workload = workload
        self._penalty_factor = penalty_factor
        '''
            1.对数据基本大小的描述信息
        '''
        batch, c_p_size = env_data.shape
        slots_size = c_p_size // 2
        _, task_size = tours.shape

        self.batch = batch
        self._slots_size = slots_size
        self._task_size = task_size

        '''
            1.sequence_key:
            排序主要参考的指标 0.01 * i 是为了参考时延,约在前面的时隙,越好用,越珍贵
            2.slot_use_seq 时隙的使用顺序
            3.work_use 完成所有任务需要的计算资源
        '''
        sequence_key = np.array([p_slots[i] for i in range(len(p_slots))
                                 for _ in range(batch)])
        # 重新改为从前往后使用时隙资源就可以了
        slot_use_seq = np.array([[j for j in range(slots_size)]
                                for i in range(batch)])
        final_slot = np.full((batch,), fill_value=-1)
        last_slot = np.full((batch,), fill_value=-1)
        for b in range(batch):
            work_use = np.sum(workload[b])
            slot_sum = 0
            for i in range(slots_size):
                '''
                    c_slots 中要选 b 组
                    后面任务的 完成顺序也要选 第 b 组
                '''
                slot_sum += c_slots[b][slot_use_seq[b][i]]
                last_slot = np.maximum(last_slot, i)
                if slot_sum >= work_use:
                    final_slot[b] = last_slot
                    break
        '''
            将整个问题改成 调用 batch 次 one_batch 首先看哪些地方需要修改成 batch
            变量
                1. self._final_slot  (1, ) to (batch, ) √
                2. self._slot_use_seq (slot_size, ) to (batch, slot_size) √
                3. sequence_key (slot_size, ) to (batch, slot_size) √
            函数:
                one_batch 由传入无参数, 变为传入的参数为要处理的批次
                
        '''

        self._slot_use_seq = slot_use_seq
        self._final_slot = final_slot
        self._now_slot = 0
        # 从前往后一直使用到哪个时隙

        self.trace = np.zeros([self.batch, task_size, slots_size])
        self.u = np.zeros((batch, ))
        self.score = np.zeros((batch, ))

    def one_batch(self, batch):
        """
            按照 slot_use_seq 和 tours 的顺序完成任务
            使用 now_slot 记录使用到 slot_use_seq 哪个时隙
            real_slot : 实际使用到 c_slots 中的哪一个
            task_occupation : 记录任务每个时隙使用了多少资源
            trace : 记录所有任务的执行记录
            score : 记录 u - c
            u : 记录总的 u
        """
        for task in self.tours[batch]:
            # 记录每个task使用了哪些时隙资源
            task_occupation = np.zeros([self._slots_size])
            while True:
                '''
                    任务完成退出任务
                    同时 score 记录(累积)
                        - alpha * t - c = 
                        - alpha * slot_use_seq[now_slot] - 
                            task_occupation.dot(p_slots[0])
                        取反,也就是
                            alpha * slot_use_seq[now_slot] + 
                            task_occupation.dot(p_slots[0])
                    同时记录总的 u (累积, 负数)
                        u = -alpha * slot_use_seq[now_slot]
                '''
                if self._workload[batch][task] < 1e-10:
                    self.score[batch] += \
                        self._penalty_factor[batch][task] * self._slot_use_seq[batch][self._now_slot] + \
                        task_occupation.dot(self._p_slots[batch])
                    self.u[batch] += - self._penalty_factor[batch][task] * \
                                     self._slot_use_seq[batch][self._now_slot]
                    break
                real_slot = self._slot_use_seq[batch][self._now_slot]
                '''
                    判断当前时隙是否可以完成任务
                    如果完成不了
                        1. 先使用本时隙的资源
                        2. 记录任务在本时隙使用了多少资源
                        3. 将本时隙资源置为0
                        4. 进入 slot_use_seq 中下一个挑选的时隙
                    如果能完成
                        1. 本时隙资源减去任务资源
                        2. 记录任务本时隙使用的资源
                        3. 将任务完成剩余资源设置为0
                '''
                if self._c_slots[batch][real_slot] < self._workload[batch][task]:
                    self._workload[batch][task] -= self._c_slots[batch][real_slot]
                    task_occupation[real_slot] += self._c_slots[batch][real_slot]
                    self._c_slots[batch][real_slot] = 0
                    self._now_slot += 1
                else:  # 当前时隙可以完成任务
                    self._c_slots[batch][real_slot] -= self._workload[batch][task]
                    task_occupation[real_slot] += self._workload[batch][task]
                    self._workload[batch][task] = 0
            self.trace[batch][task] = task_occupation
        '''
            一个批次处理完毕,需要重新设置 self._now_slot
        '''
        self._now_slot = 0

    def punish(self, batch):
        """
        如果任务无法完成,对应批次设置惩罚
            u 和 trace 初始化自动为 0
        :param batch: 第多少批
        :return: None
        """
        self.score[batch] = 5000.


    def cal(self, batch):
        """
        任务有可行方案,计算具体的可行方案
            将self._final_slot[batch] 以后的使用slot变为 np.iinfo(np.int32).max
        :param batch: 对应的批次
        :return: None
        """
        for i in range(self._final_slot[batch] + 1,
                       len(self._slot_use_seq[batch])):
            self._slot_use_seq[batch][i] = np.iinfo(np.int32).max

        self._slot_use_seq[batch].sort()
        self.one_batch(batch)

    def greed_score(self):

        '''
            循环 batch
                不能完成的任务直接continue
                能完成的任务才会改变长度
        '''
        for b in range(self.batch):

            if self._final_slot[b] == -1:  # 不能完成任务
                self.punish(b)  # 设定惩罚
                continue
            '''
                可以完成任务,先截断 slot_use_seq
                并且 slot_use_seq 必须是从前到后的使用顺序
                也就是说需要再对 slot_use_seq 做一次排序
            '''
            self.cal(b)

        '''
            score (1, ) to (batch, 1)
            u     (1, ) to (batch, 1)
            trace (task_size, slot_size) to (batch, task_size, slot_size)
        '''
        # return torch.from_numpy(d).to(device), u, trace  # 一定要返回cuda数据（gpu)


if __name__ == "__main__":
    inputs = [torch.tensor([[[591.4531, 22.4923],
                             [218.5055, 17.7651],
                             [466.0721, 13.7677],
                             [284.1283, 8.1004],
                             [195.4626, 24.7843],
                             [306.9142, 21.1846],
                             [599.2676, 14.5494],
                             [0.0000, 0.0000],
                             [0.0000, 0.0000],
                             [0.0000, 0.0000]]]),
              torch.tensor([[1.4635e+02, 5.1574e+01, 7.9357e+01, 1.4067e+02, 6.3253e+01, 9.3808e+01,
                             6.7542e+01, 7.7266e+01, 1.2939e+02, 9.8069e+01, 9.8686e+01, 7.3700e+01,
                             8.4672e+01, 1.4061e+02, 7.0344e+01, 1.4020e+02, 7.8021e+01, 5.3114e+01,
                             5.0951e+01, 1.4496e+02, 7.7031e+01, 1.2481e+02, 9.9813e+01, 7.7184e+01,
                             7.2463e+01, 9.2338e+01, 1.3529e+02, 5.7712e+01, 1.3760e+02, 9.6516e+01,
                             5.4414e-01, 9.9070e-01, 6.6059e-01, 6.0580e-01, 2.0286e-01, 8.1328e-01,
                             7.4790e-01, 1.7200e-01, 9.3386e-01, 7.5517e-01, 2.7740e-01, 8.8028e-01,
                             2.3666e-01, 6.5084e-01, 2.8878e-01, 6.5283e-01, 2.6687e-02, 4.9296e-01,
                             8.6338e-01, 1.2337e-01, 4.3695e-02, 3.1412e-01, 4.2735e-01, 7.2491e-01,
                             6.7175e-01, 8.8491e-01, 6.2024e-01, 2.1663e-01, 5.6574e-01, 9.7774e-01]])]
    tours = torch.tensor([[1, 9, 4, 0, 5, 6, 8, 7, 3, 2]], device='cuda:0')
    inputCopy = copy.deepcopy(inputs)
    greedy = Greedy(inputs, tours)
    greedy.greed_score()
