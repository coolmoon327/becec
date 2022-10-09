'''
Author: your name
Date: 2021-12-25 22:25:04
LastEditTime: 2022-03-17 10:57:27
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /TSP_DRL_PtrNet-master/search.py
'''
import copy
from math import pi

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
from datetime import datetime
from env.becec.stage_two.actor import PtrNet1
from env.becec.stage_two.Sequencing import Sequencing
from env.becec.stage_two.Greedy import Greedy


def sampling(cfg, env, test_input):
    """
    返回对比算法的排序
    :param cfg:
    :param env:
    :param test_input:
    :return: score, u, trace
    """
    '''
        1. 找出对应的序列应该长什么样子
    '''
    seq = Sequencing(test_input)
    seq.sequncing()
    # 先在这个位置尝试添加对 dp 修改的版本, 输入尽量和 Greedy 完全一致,
    # 方便后续直接使用 seq 的输出部分
    greedy = Greedy(test_input,
                    torch.tensor(seq.res).unsqueeze(0))
    greedy.greed_score()

    dp = DP(test_input, torch.tensor(seq.res).unsqueeze(0))
    dp.score()

    # return greedy.score, greedy.u, greedy.traceInfo
    return dp.cost, dp.u, dp.traceInfo


def active_search(cfg, env, test_input, log_path=None):
    '''
    active search updates model parameters even during inference on a single input
    test input:(city_t,xy)
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    task_data, env_data = test_input
    # task_data.to(device)
    # env_data.to(device)
    batch, _ = env_data.size()
    date = datetime.now().strftime('%m%d_%H_%M')
    random_tours = env.stack_random_tours(batch)  # 相当于只是加了一层
    baseline = env.seq_score(test_input, random_tours)
    l_min = torch.min(baseline)  # l_min 记录的是最小成本吗，还是需要记录最大成本，有必要使用吗

    act_model = PtrNet1(cfg)
    if os.path.exists(cfg.act_model_path):
        act_model.load_state_dict(
            torch.load(cfg.act_model_path, map_location=device))

    if cfg.optim == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=cfg.lr)

    act_model = act_model.to(device)

    for i in tqdm(range(cfg.steps)):
        '''
        - page 6/15 in papar
        we randomly shuffle the input sequence before feeding it to our pointer network. 
        This increases the stochasticity of the sampling procedure and leads to large improvements in Active Search.
        '''
        task_data, env_data = test_input
        task_data = env.shuffle(task_data)  # 将task_data任务前后顺序打乱
        shuffle_inputs = TensorDataset(task_data, env_data)
        inputs = DataLoader(shuffle_inputs, batch_size=batch, shuffle=False)
        for j, shuffle_inputs in enumerate(inputs):
            # task_data, env_data = shuffle_inputs
            pred_shuffle_tours, neg_log = act_model(shuffle_inputs, device)
            # shuffle_inputs = task_data
            task_data, env_data = test_input
            test_task_input = task_data  # fixme:对各个位置的理解
            pred_tours = env.back_tours(pred_shuffle_tours, shuffle_inputs,
                                        test_task_input)

            l_batch = env.seq_score(test_input, pred_tours)

            index_lmin = torch.argmin(l_batch)  # 最小消耗也就是最好的
            if torch.min(l_batch) != l_batch[index_lmin]:
                raise RuntimeError
            if l_batch[index_lmin] < l_min:
                best_tour = pred_tours[index_lmin]
                print('update best tour, min l(%1.3f -> %1.3f)' % (
                    l_min, l_batch[index_lmin]))
                l_min = l_batch[index_lmin]

            adv = l_batch - baseline
            act_optim.zero_grad()
            act_loss = torch.mean(adv * neg_log)
            '''
            adv(batch) = l_batch(batch) - baseline(batch)
            mean(adv(batch) * neg_log(batch)) -> act_loss(scalar) 
            '''
            act_loss.backward()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1.,
                                     norm_type=2)
            act_optim.step()
            baseline = baseline * cfg.alpha + (1 - cfg.alpha) * torch.mean(
                l_batch, dim=0)
            print(
                'step:%d/%d, actic loss:%1.3f' % (i, cfg.steps, act_loss.data))

        if cfg.islogger:
            if i % cfg.log_step == 0:
                if log_path is None:
                    log_path = cfg.log_dir + 'active_search_%s.csv' % (
                        date)  # cfg.log_dir = ./Csv/
                    with open(log_path, 'w') as f:
                        f.write('step,actic loss,minimum distance\n')
                else:
                    with open(log_path, 'a') as f:
                        f.write('%d,%1.4f,%1.4f\n' % (i, act_loss, l_min))
    return 0


class DP:
    def __init__(self, inputs, tours):
        """
            :param inputs: (batch, city_t, 2 + slots * 2)
                workload, alpha, c, p
            :param tours: (batch, city_t), predicted orders
        """

        data = copy.deepcopy(inputs)
        data = data.numpy()
        _, task_size = tours.size()
        # 这里的 indices_or_sections = [2] 永远不用变， 因为只是最后一个维度， 而一个任务永远对应 22
        task_data, env_data = np.split(data, axis=2, indices_or_sections=[2])
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
        self.unfinished_tasks = np.full_like(tours, -1)

        '''
            1.sequence_key:
            排序主要参考的指标 0.01 * i 是为了参考时延,约在前面的时隙,越好用,越珍贵
            2.slot_use_seq 时隙的使用顺序
            3.work_use 完成所有任务需要的计算资源
        '''
        sequence_key = np.array([p_slots[i] for i in range(len(p_slots))
                                 for _ in range(batch)])
        slot_use_seq = np.array([[j for j in range(slots_size)]
                                 for i in range(batch)])
        final_slot = np.full((batch,), fill_value=-1)
        last_slot = np.full((batch,), fill_value=-1)
        for b in range(batch):
            count = task_size - 1
            while count >= 0:
                # 默认被 -1 标记的是无法完成的任务
                work_use = np.sum(workload[b][[tours[b][i] for i in range(
                    len(tours[b])) if tours[b][i] != -1]])
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
                else:  # for 循环非正常退出, 可以跳出 while 语句了
                    # 如果无法完成任务, 那么 tours[b][count] 需要被设置为 -1
                    if final_slot[b] == -1:
                        self.unfinished_tasks[b][task_size - 1 - count] = \
                            tours[b][count]
                        tours[b][count] = -1

                # 已经确定当前的 count 完成剩下的 tasks
                if tours[b][count] != -1:
                    break
                count -= 1
        # 所以后面创建的也不应该是 task_size 对应的大小(虽然不影响, 但是害怕对结束条件
        # 的处理有所不同)
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
        self.u = np.zeros((batch,))
        # 添加一个字典记录 trace, 完成了的任务序号, 没有完成的任务序号
        # trace: (batch, task_size, slots_size)
        # finish_tasks: (batch, 已完成任务的编号)
        # unfinished_tasks: (batch, 未完成任务的编号)
        # 这里的 tours 不能在前面完全统计出来, 所以需要在后面进行一次处理
        self.traceInfo = {'trace': self.trace,
                          'tours': tours}

    def score(self):
        """动态规划部分的整体打分, 需要返回 u, c, trace"""
        # 首先需要把整体的trace给先算出来其它的都好说

        # 1. 预先判断可以完成多少任务, 并且添加最后一个时隙, 保证所有任务都是可以完成的
        #    这一步在初始化的时候已经完成了, 进入此处的 tours 中包含的都是可以执行的任务

        # 2. dp计算
        # 思路
        # 2.1 优先使用的还是时隙比较便宜的资源, 可以考虑的时隙资源不包含最后一个时隙
        # 2.2 本做法可以认为前面一个任务占用的时隙, 本任务一定是不会占用的, 这样的话就可以使用
        #     之前dp的完成方法继续实现
        INF = 44444444  # fixme: 整体模块的可使用性不能完全确定

        # 这里的 indices_or_sections 是不是应该要变为数据的实际长度
        tours = self.tours

        c_slots = self._c_slots
        c_sum_slots = np.cumsum(c_slots, axis=1)  # 对最后一个维度进行累加
        batch = self.batch

        # c_slots, c_sum_slots, env_data, p_slots 的有效数据从0开始使用
        dp = np.full((batch, self._task_size, self._slots_size),
                     fill_value=-INF,
                     dtype=np.float32)
        real_worker = np.full((batch, self._slots_size, self._slots_size),
                              fill_value=0,
                              dtype=np.int)  # (batch, slots_size, slots_size) i： 多少批次 j：以 j 为 last_t 方案 k：具体方案的具体实施方法
        total_utility = np.full((batch, self._task_size, self._slots_size),
                                fill_value=0,
                                dtype=np.float32)  # 和 dp 模块等大小,记录整体的 utility
        # worker_trace = np.full((batch, task_size, slots_size), fill_value=0, dtype=np.int) # 完整记录每个时间点上每个任务分配了多少资源

        workload = self._workload
        penalty_factor = self._penalty_factor
        for b in range(batch):
            remain_cpu = [0 for _ in range(self._slots_size)]
            for i in range(self._task_size):  # 并不是处理所有任务, 而是处理非 -1 的任务,
                # 所以阶段的位置也需要特殊处理
                cur_task_index = tours[b][i]
                if cur_task_index == -1:  # 后面的任务一定全部都无法处理了
                    break

                if i == 0:  # 特殊处理第一个任务
                    j = self._slots_size - 1
                    while j >= 0:  # 从后往前遍历
                        if (c_sum_slots[b][j] >= workload[b][cur_task_index]):
                            worker = np.full((self._slots_size,), fill_value=0,
                                             dtype=np.int)  # 记录可行的一个方案, 针对 i 任务的可行方案
                            cost = 0
                            utility = 0
                            last_t = 0
                            rank = self._p_slots[b][
                                   :j + 1].argsort()  # 按照rank从左到右使用时隙
                            rank = iter(rank)
                            workload_temp = workload[b][cur_task_index]
                            p_min_slot = -1
                            while (workload_temp > 0):
                                p_min_slot = next(rank, -1)
                                last_t = max(last_t, p_min_slot)
                                workload_temp = workload_temp - c_slots[b][
                                    p_min_slot]
                                cost = cost + c_slots[b][p_min_slot] * \
                                       self._p_slots[b][p_min_slot]
                                worker[p_min_slot] = c_slots[b][
                                    p_min_slot]  # 某一个方案中使用了某一个时隙的资源，有多少
                            utility = -last_t * penalty_factor[b][
                                cur_task_index]
                            if p_min_slot == last_t and workload_temp < 0:  # 最后使用的时隙是 last_t 并且有资源的剩余
                                worker[last_t] = worker[
                                                     last_t] + workload_temp  # 第一个任务的 last_t
                                cost = cost + workload_temp * self._p_slots[
                                    b][last_t]
                                workload_temp = 0
                                if utility - cost > dp[b][0][
                                    last_t]:  # 最后的时隙无剩余完成任务
                                    real_worker[b][last_t] = copy.deepcopy(
                                        worker)  # 刚开始完全复制 是没有问题的, 而且一定不需要考虑 last_t 重复的问题
                                    dp[b][0][last_t] = utility - cost
                                    total_utility[b][0][
                                        last_t] = utility  # 第一个任务, 先记录所有的 utility
                                    remain_cpu[last_t] = -workload_temp
                            if p_min_slot == -1:  # 初始任务负载为0: 如果进入这里 默认的是 last_t 就是0
                                remain_cpu[last_t] = c_slots[b][
                                    last_t]  # 如果说使用的成本很高呢？一定要牢记使用量必定是 0
                                worker[last_t] = 0  # 同样的可行方案中 有一个的使用量是 0
                                total_utility[b][0][last_t] = 0
                                real_worker[b][last_t] = copy.deepcopy(worker)
                                workload_temp = 0
                            # dp[b][0][last_t] = max(dp[b][0][last_t], (utility - cost)) # total u - c
                            # 如果选择的是 u - c 则无理由的替换掉 real_worker[b][last]
                            if p_min_slot != -1:  # 当前任务的工作量大于 0
                                cost = cost + workload_temp * self._p_slots[b][
                                    p_min_slot]  # 有一部分的 c_slots[b][p_min_slot] 是没有用完的
                            if utility - cost > dp[b][0][
                                last_t]:  # 完成任务的最后一个时隙不是 last_t 是在中间的某一个位置
                                if p_min_slot != -1:
                                    worker[p_min_slot] = worker[
                                                             p_min_slot] + workload_temp
                                real_worker[b][last_t] = copy.deepcopy(
                                    worker)  # 刚开始完全复制 是没有问题的, 而且一定不需要考虑 last_t 重复的问题
                                dp[b][0][last_t] = utility - cost
                                total_utility[b][0][
                                    last_t] = utility  # 第一个任务, 先记录所有的 utility
                                remain_cpu[last_t] = 0
                            # last_t 包括以后的时隙全部都不需要管了
                            j = last_t - 1
                        else:
                            j -= 1

                    largest_num_left = -INF  # 记录左侧的最大值
                    slots_posible_stratage = np.array([],
                                                      dtype=np.int)  # 记录可行方案的时刻
                    for j in range(self._slots_size):  # 清除右边比左边小的值
                        if dp[b][0][j] <= largest_num_left:
                            dp[b][0][j] = -INF
                            total_utility[b][0][j] = 0  # 不具备可行方案
                            real_worker[b][j] = [0 for _ in range(
                                self._slots_size)]  # 如果说没有可行方案,
                            # 将last_t对应的方案全部置零
                        else:
                            largest_num_left = dp[b][0][j]
                            slots_posible_stratage = np.append(
                                slots_posible_stratage,
                                j)  # 对应的位置才需要查询是否有remain的cpu资源, 下面的对应方法一定是将第一层的worker 做过清除
                else:  # i > 0
                    largest_num_left = -INF  # 记录左侧的最大值
                    remain_temp = [0 for _ in range(
                        self._slots_size)]  # 从此处开始使用
                    # remain_temp，作用是记录完成这个任务过程中有哪些时隙是最后使用的并且有可以使用的剩余资源
                    worker_temp = np.full((batch, self._slots_size,
                                           self._slots_size),
                                          fill_value=0,
                                          dtype=np.int)  # 记录本轮任务完成时的完整使用轨迹
                    for j in slots_posible_stratage:  # 可能的左端点
                        right = self._slots_size
                        while (
                                right > j):  # 每进入一次， remain_cpu 都应该和上一个任务结束时的完全一致
                            cost = 0
                            utility = 0
                            if remain_cpu[
                                j] > 0:  # 上面的所有任务在j完成，并且在j时隙还有剩余的资源可以使用
                                if (c_sum_slots[b][right - 1] - c_sum_slots[b][
                                    j] + remain_cpu[j]) >= workload[b][
                                    cur_task_index]:
                                    worker = np.full((self._slots_size,),
                                                     fill_value=0, dtype=np.int)
                                    last_t = 0
                                    rank = self._p_slots[b][j + 1:
                                                            right].argsort()
                                    rank = rank + j + 1  # 可以从 j 时隙开始使用
                                    rank = iter(rank)
                                    workload_temp = workload[b][cur_task_index]
                                    p_min_slot = -1
                                    remain_cpu_term = copy.deepcopy(
                                        remain_cpu)  # remain_cpu_term上一个任务结束后有多少可以使用的资源，可以在不同的起始端点重复使用remain_cpu_term，对它进行修改相当于一个标志，用来辅助修改cpu_temp
                                    if workload_temp <= 0:  # 跳过workload为0的情况
                                        if dp[b][i - 1][j] > dp[b][i][
                                            j]:  # 涉及对 remain_temp[j] 的重复赋值，只有当 dp 的取值更加时，对 remain_temp[j] 重复赋值
                                            dp[b][i][j] = dp[b][i - 1][j]
                                            total_utility[b][i][j] = \
                                                total_utility[b][i - 1][j] - j * \
                                                penalty_factor[b][
                                                    cur_task_index]  # 整体的utility会增加一个在  j 时刻结束的 分量
                                            remain_temp[j] = remain_cpu[
                                                j]  # 最后一个时隙完成了任务，并且还有剩余的资源
                                            worker_temp[b][j] = real_worker[b][
                                                j]  # 对最后一个时隙的重复
                                        # worker = copy.deepcopy(real_worker[b][j]) # 即使仍然需要赋值，需要做的事情是 real_worker[b][j] = real_worker[b][j]
                                    else:  # 如果说 workload 非 0
                                        workload_temp -= remain_cpu_term[
                                            j]  # 首先使用一部分 cpu 资源, 但是在while循环中会重复使用 remain_cpu 的资源, 因此在排序的时候应该和remain_cpu为0的方式相同考虑
                                        cost = cost + remain_cpu_term[j] * \
                                               self._p_slots[b][
                                                   j]  # 首先得默认 remain_cpu 使用完毕
                                        worker[j] += remain_cpu_term[
                                            j]  # 默认使用 j 时隙的所有剩余 cpu 资源, 首先使用了 remain 的资源，但是问题是以62结尾的应该没有剩余的资源了，为什么还要继续使用呢
                                        if workload_temp < 0:  # 依旧无法使用完 remain_cpu[j] ，j 依旧是最后的一个时隙
                                            remain_cpu_term[
                                                j] = -workload_temp  # 给本轮使用
                                            worker[
                                                j] += workload_temp  # 剩余 cpu 资源多的退回去
                                            last_t = j
                                            utility = -last_t * \
                                                      penalty_factor[b][
                                                          cur_task_index]
                                            cost = cost + workload_temp * \
                                                   self._p_slots[b][j]
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) >= dp[b][i][
                                                last_t]:  # 新的分配方案更优
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                                remain_temp[
                                                    j] = -workload_temp  # remain_temp[j] 是否会重复赋值，应该选取其中的哪一个？应该选取dp数值更大的那一个
                                                worker_temp[b][last_t] = \
                                                    real_worker[b][
                                                        j] + worker  # 上次结束使用的时隙是 j , 现在还是在使用 last_t的时隙
                                            workload_temp = 0  # 资源反还完毕，刚好完成任务
                                        while (
                                                workload_temp > 0):  # 使用完了上个任务剩余时隙的资源,接着还需要找
                                            p_min_slot = next(rank, -1)
                                            last_t = max(last_t, p_min_slot)
                                            workload_temp = workload_temp - \
                                                            c_slots[b][
                                                                p_min_slot]
                                            cost = cost + c_slots[b][
                                                p_min_slot] * self._p_slots[b][
                                                       p_min_slot]
                                            worker[p_min_slot] = c_slots[b][
                                                p_min_slot]  # 记录在某一个时隙使用了多少资源
                                        utility = -last_t * penalty_factor[b][
                                            cur_task_index]
                                        if p_min_slot == last_t and workload_temp < 0:  # 任务完成了，并且最后的一个时隙有剩余cpu的情况
                                            worker[last_t] = worker[
                                                                 last_t] + workload_temp  # 最后一个时隙多使用的资源不能记录在worker
                                            cost = cost + workload_temp * \
                                                   self._p_slots[b][
                                                       last_t]  # 发现有一部分资源没有使用完
                                            workload_temp = 0  # 完成该任务
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) > dp[b][i][
                                                last_t]:  # 只在这种情况下进行重新赋值是否正确呢？ 我的答案应该是正确的
                                                worker_temp[b][
                                                    last_t] = worker + \
                                                              real_worker[b][
                                                                  j]  # 上一次的 j 还有残留 不用考虑和下面的 if 重叠的状态
                                                remain_temp[
                                                    last_t] = -workload_temp  # 这种情况下需要对remain_temp 进行更新
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                        if p_min_slot != -1:  # 当前任务的工作量大于 0
                                            cost = cost + workload_temp * \
                                                   self._p_slots[b][p_min_slot]
                                        if ((utility - cost) + dp[b][i - 1][
                                            j]) > dp[b][i][
                                            last_t]:  # 这种状态下说明最后完成的 slot 没有剩余的资源，仅仅是对 dp 进行一次更新
                                            if p_min_slot != -1:
                                                worker[
                                                    p_min_slot] += workload_temp  # 多使用了一部分的 c_slots 资源
                                            worker_temp[b][last_t] = worker + \
                                                                     real_worker[
                                                                         b][
                                                                         j]  # 上一次的 j 还有残留
                                            dp[b][i][last_t] = (
                                                    (utility - cost) +
                                                    dp[b][i - 1][j])
                                            total_utility[b][i][last_t] = \
                                                total_utility[b][i - 1][
                                                    j] + utility
                                            remain_temp[last_t] = 0
                                    right = last_t
                                else:
                                    right -= 1
                            else:  # remain_cpu = 0
                                if (c_sum_slots[b][right - 1] - c_sum_slots[b][
                                    j]) >= workload[b][
                                    cur_task_index]:  # 从时隙 j + 1开始能满足工作负载要求
                                    worker = np.full((self._slots_size,),
                                                     fill_value=0, dtype=np.int)
                                    last_t = 0
                                    rank = self._p_slots[b][j + 1:
                                                            right].argsort()
                                    rank = rank + j + 1
                                    rank = iter(rank)
                                    workload_temp = workload[b][cur_task_index]
                                    remain_cpu_term = copy.deepcopy(remain_cpu)
                                    if workload_temp <= 0:  # 当前的任务需要的资源量是 0 , 最后一个完成的时隙是 j, 且完成了任务
                                        if dp[b][i - 1][j] > dp[b][i][j]:
                                            dp[b][i][j] = dp[b][i - 1][j]
                                            total_utility[b][i][j] = \
                                                total_utility[b][i - 1][j] - j * \
                                                penalty_factor[b][
                                                    cur_task_index]
                                            remain_temp[j] = remain_cpu[j]
                                            worker_temp[b][j] = real_worker[b][
                                                j]
                                    else:  # 当前任务的资源需求量大于0
                                        p_min_slot = -1
                                        while (workload_temp > 0):
                                            p_min_slot = next(rank, -1)
                                            last_t = max(last_t, p_min_slot)
                                            workload_temp = workload_temp - \
                                                            c_slots[b][
                                                                p_min_slot]
                                            worker[p_min_slot] = c_slots[b][
                                                p_min_slot]  # 记录下某一个时隙使用了具体多少资源
                                            cost = cost + c_slots[b][
                                                p_min_slot] * self._p_slots[b][
                                                       p_min_slot]
                                        utility = -last_t * penalty_factor[b][
                                            cur_task_index]
                                        if p_min_slot == last_t and workload_temp < 0:  # 在最后的一个时隙完成了任务，并且需要更新 remain_temp 和 dp
                                            worker[last_t] = worker[
                                                                 last_t] + workload_temp  # 实际环境少使用的资源需要吐出来
                                            cost = cost + workload_temp * \
                                                   self._p_slots[b][
                                                       p_min_slot]  # 少使用的花费
                                            workload_temp = 0
                                            if ((utility - cost) + dp[b][i - 1][
                                                j]) >= dp[b][i][last_t]:
                                                remain_temp[
                                                    last_t] = -workload_temp
                                                dp[b][i][last_t] = (
                                                        (utility - cost) +
                                                        dp[b][i - 1][j])
                                                total_utility[b][i][last_t] = \
                                                    total_utility[b][i - 1][
                                                        j] + utility
                                                worker_temp[b][last_t] = (
                                                        worker +
                                                        real_worker[b][j])
                                        if p_min_slot != -1:  # 当前任务的工作量大于 0
                                            cost = cost + workload_temp * \
                                                   self._p_slots[b][p_min_slot]
                                        if ((utility - cost) + dp[b][i - 1][
                                            j]) >= dp[b][i][
                                            last_t]:  # 做的事情完成任务的时隙不是最后的时隙，在这种情况下，只需要更新 dp, remain_cpu 不需要更新, workload_temp 同样可能是小于 0 的
                                            if p_min_slot != -1:
                                                worker[
                                                    p_min_slot] += workload_temp
                                            dp[b][i][last_t] = (
                                                    (utility - cost) +
                                                    dp[b][i - 1][
                                                        j])  # 上一个任务在 j 时隙已经完成了
                                            total_utility[b][i][last_t] = \
                                                total_utility[b][i - 1][
                                                    j] + utility
                                            worker_temp[b][last_t] = (
                                                    worker + real_worker[b][
                                                j])  # 全新的数, j 是上一个任务完成的时隙
                                            remain_temp[
                                                last_t] = 0  # 需要清空最后一个时隙的资源
                                    right = last_t
                                else:
                                    right -= 1
                    slots_posible_stratage = np.array([],
                                                      dtype=np.int)  # 记录可行方案的时刻
                    for j in range(self._slots_size):  # 清除右边比左边小的值
                        if dp[b][i][j] <= largest_num_left:
                            dp[b][i][j] = -INF
                            total_utility[b][i][j] = 0
                            worker_temp[b][j] = [0 for _ in range(
                                self._slots_size)]
                            remain_temp[j] = 0
                        else:
                            largest_num_left = dp[b][i][j]
                            slots_posible_stratage = np.append(
                                slots_posible_stratage, j)
                    remain_cpu = copy.deepcopy(remain_temp)  # 此处更新位置没有错
                    real_worker = copy.deepcopy(
                        worker_temp)  # 此处还需要更新 real_worker 的数据
        d = np.full((batch,), fill_value=-INF, dtype=np.float32)
        u = np.full((batch,), fill_value=0, dtype=np.float32)
        actor_source = np.full((batch, self._slots_size), fill_value=0,
                               dtype=np.int)
        for b in range(batch):
            d[b] = -np.max(dp[b][self._task_size - 1])
            max_index = np.argmax(
                dp[b][self._task_size - 1])  # 无可行方案时如何记录, 无可行方案时, d[b] 返回的是
            actor_source[b] = copy.deepcopy(
                real_worker[b][max_index])  # 如果出现没有可行方案怎么办？是否应该返回无法执行
            u[b] = total_utility[b][self._task_size - 1][max_index]
            if (-np.max(dp[b][self._task_size - 1]) == INF):
                # 这里说明就是没有可行方案, 优先将所有的任务都默认完成不了, 也就是可以完成任务的数量为0
                # 同时令未完成的任务顺序变为0, 1, 2, 3, .....
                # 因为 tours 在最前面已经有了一次筛选了, 所以不完成的概率会低一点
                # 那么在开头的位置应该要想办法处理 tours 中 -1 的任务
                tours[b][:] = -1  # 表示这些任务没有完成, 后面会自动进行统计, 已经有了在前面的一次预筛选了
                d[b] = 0
                u[b] = 0  # 如果没有可行方案, 则返回的utility为0
        trace = np.full((batch, self._task_size, self._slots_size),
                        fill_value=0,
                        dtype=np.int)
        for b in range(batch):
            slot = 0  # 从 0 时隙 遍历到第 99 时隙
            slot_over_flag = False
            for i in range(len(tours[b])):  # 第 i 个任务
                cur_task_index = tours[b][i]
                workload_temp = workload[b][cur_task_index]
                while workload_temp > 0:  # 当前任务没有完成
                    if actor_source[b][
                        slot] > 0:  # 有时隙资源 如果说没有可行方案，还需要管 trace 的计算吗？如何判断没有可行方案？ 根据打分大小吗？
                        trace[b][i][slot] = actor_source[b][slot]
                        workload_temp -= actor_source[b][slot]
                    slot += 1  # 每判断一次递进到下一个 slot
                    if slot == self._slots_size:
                        slot_over_flag = True
                        break
                if slot_over_flag:
                    break

        # print('seq_score down')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.trace = trace
        self.u = u
        self.cost = d
        self.traceInfo['tours'] = tours
        self.traceInfo['trace'] = trace[:, tours[0][:], :]

        # 3. 后续的资源量往前补齐, 改变 trace 最后一个时隙的样子
