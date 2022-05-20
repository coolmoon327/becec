"""
    实现对比算法
"""
import time

import numpy as np
import torch
import copy


class Sequencing(object):
    def __init__(self, seq_input):
        """
        :param seq_input:
                        task_data (batch, task_size, 2 + slots * 2)
                        前面的两个属性分别是 workload alpha
                        需要将它给取出来
        v 表示价值
        w 表示权重
        trace 表示选中了哪些任务
        upper 表示 Universal Sequencing on an Unreliable Machine
                中集合大小的上界
        """
        mixData = copy.deepcopy(seq_input)
        mixData = mixData.numpy()
        task_data, env_data = np.split(mixData, axis=2, indices_or_sections=[2])
        self.data = task_data
        self.length = len(seq_input[0])
        self.trace = np.empty((0, 1), dtype=np.int8)
        self.v = np.array([self.data[0][_][0] for _ in range(self.length)])
        self.w = np.array([self.data[0][_][1] for _ in range(self.length)]).dot(1000)
        self.w = np.ceil(self.w).astype(np.int32)
        self.upper = np.ceil(np.log2(sum(self.w))).astype(np.int32)
        self.res = None

    def sequncing(self):
        """
            dp 的计算过程
            1. 获得每一个小集合的元素
            2. 从前往后,累计所有的集合元素
                2 ** 0 到 2 ** upper
            3. 从后往前求差集, 打印 setdiff1d(x, y)
                元素在 x 中,但不在 y 中
            res: 任务排序
        """
        '''
            1. 获得每一个小集合的元素
        '''
        s = []
        for i in range(0, self.upper + 1):
            self.solution(2 ** i, v=self.v, w=self.w)
            s.append(self.trace)
        '''
            2. 从前往后,累计所有的集合元素
                2 ** 0 到 2 ** upper
        '''
        for i in range(1, self.upper + 1):
            s[i] = np.union1d(s[i], s[i - 1])
        '''
            3. 从后往前求差集, 打印 setdiff1d(x, y)
                元素在 x 中,但不在 y 中
        '''
        res = np.empty((0, 1), dtype=np.int8)
        for i in range(self.upper, 0, -1):
            res = np.append(res, np.setdiff1d(s[i],
                                              s[i - 1]))
        self.res = res

    def traceBack(self, c, v, w, p):
        """
        路径回溯,找到最优组合
        :param c: 背包的容量为 c
        :param v: 物品的价值列表
        :param w: 物品的重量列表
        :param p: dp 的实际过程
        :return:
        """
        '''
            k: 记录初始的容量
            1. 如果发现 result[i][j] = result[i + 1][j - w[i]] + v[i] 
                则该物品一定被放置
            2. 同时还需要记录最后的一个任务是否被使用
                记录时需要注意下标从 1 开始
                将下标改为从 0 开始


        '''
        k = c
        trace = np.empty((0, 1))
        for i in range(len(p) - 1):
            if p[i][k] == p[i + 1][k - w[i]] + v[i]:
                k = k - w[i]
                trace = np.append(trace, i)

        if p[len(p) - 1][k] == v[len(p) - 1]:
            trace = np.append(trace, len(p) - 1)

        print(trace)

    def basicSolution(self, c, v, w):
        """
        :param c: 背包的容量为 c
        :param v: 物品的价值列表
        :param w: 物品的重量列表
        :return: result 有限容量下的最大价值
        """
        '''
            n 代表物品个数
            m 代表上限容量, 加1是为了避免判断时数组越界
            1. 选择是否选取最后一个物品
            2. 从倒数第二个开始向第一个递归0
        '''
        n = len(v)
        m = c + 1
        result = np.zeros((n, m))
        for i in range(m):
            result[n - 1][i] = v[n - 1] if i >= w[n - 1] else 0
        for i in range(n - 2, -1, -1):
            for j in range(m):
                if j < w[i]:
                    result[i][j] = result[i + 1][j]
                else:
                    result[i][j] = np.maximum(result[i + 1][j],
                                              result[i + 1][j - w[i]] + v[i])
        self.traceBack(c, v, w, result)
        return result[0][c]

    def back(self, v, w, p, head):
        """
        追溯最优组合
        :param v: 物品的价值列表
        :param w: 物品的重量列表
        :param p: 跳跃点的记录
        :param head: 下一个跳跃点
        :return: None
        """
        self.trace = np.empty((0, 1), dtype=np.int8)
        k = head[0] - 1
        n = len(w)
        for i in range(1, n + 1):
            left = head[i + 1]
            right = head[i] - 1
            for j in range(left, right + 1):
                if p[j][0] + w[i - 1] == p[k][0] and \
                        p[j][1] + v[i - 1] == p[k][1]:
                    k = j
                    self.trace = np.append(self.trace, i - 1)
                    break
        '''
            self.trace 记录具体选中哪些任务
            每执行一次 solution self.trace 都会更新
        '''

    def solution(self, c, v, w):
        """
        :param c: 背包的容量为 c
        :param v: 物品的价值列表
        :param w: 物品的重量列表
        :return: result 有限容量下的最大价值
        """
        '''
            p 记录的是整个动态规划工程中的跳跃点
                1. 初始点是必记录的
                2. 后续接入的点是 重量 和 价值
            n: 物品个数
        '''
        p = np.zeros((10000, 2))
        p[0][0] = p[0][1] = 0
        n = len(v)
        left = 0
        right = 0
        next = 1
        head = np.zeros((n + 2,), dtype=np.int32)
        head[n + 1] = 0
        head[n] = 1

        for i in range(n - 1, -1, -1):
            k = left
            for j in range(left, right + 1):
                if p[j][0] + w[i] > c:
                    break
                nw = p[j][0] + w[i]
                nv = p[j][1] + v[i]
                '''
                    1. 放入比nw小的跳跃点,重量小的价值无论大小
                    2. 重量相等,取价值大的跳跃点
                    3. 去除比更新点重量大而价值小的点
                        由于是每一次更新完之后结果都是重量和价值都是递增的跳跃点
                        排列,一旦出现价值超过当前的点,
                        那后续的点的价值一定都是超过的
                    k : 变动的左端点
                    j : 变动的右端点
                    next : 跳动点记录
                '''
                while k <= right and p[k][0] < nw:
                    p[next][0] = p[k][0]
                    p[next][1] = p[k][1]
                    k += 1
                    next += 1
                '''
                    退出 while 时
                    k > right or p[k][0] >= nw
                    不考虑 k > right 已经遍历完一个left to right
                        1. 如果说 p[k][0] == nw, 说明重量相同,
                            如果此时 p[k][1] > nv 说明重量相同,
                            而价值更大, 因此将跳跃点的更新价值增大
                            保证是一个递增的序列
                        2. 如果说 p[k][0] > nw, 说明后面的重量更大
                            当前查看left to right的右端点的价值比
                            最后一个跳动点的都要大,说明右端点的确可以
                            作为下一个跳动点
                '''
                if k <= right and p[k][0] == nw:
                    if p[k][1] > nv:
                        nv = p[k][1]
                    k += 1

                if nv > p[next - 1][1]:
                    p[next][0] = nw
                    p[next][1] = nv
                    next += 1
                '''
                    p[k][0] >= nw
                    p[k][1] <= nv
                    如果到此处,说明后面的重量大但是
                    价值不高,因此不是跳跃点
                    可以掠过,保证 k > right
                        含义是 left > right
                '''
                while k <= right and p[k][1] <= nv:
                    k += 1

            '''
                放入后续的点
            '''
            while k <= right:
                p[next][0] = p[k][0]
                p[next][1] = p[k][1]
                k += 1
                next += 1
            left = right + 1
            right = next - 1
            head[i] = next
        self.back(v, w, p, head)
        '''
            return p[next - 1][1] 返回最大值
            return self.back(v, w, p, head) 返回具体是哪些任务被选中
        '''
        return p[next - 1][1]


if __name__ == "__main__":
    data = [torch.tensor([[[2.1029e-02, 7.6808e+00],
                           [7.5053e-02, 2.2964e+01],
                           [9.4003e-02, 1.1500e+01],
                           [3.0922e-02, 1.5251e+01],
                           [6.2148e-02, 2.3271e+01],
                           [7.6808e-02, 2.2580e+01],
                           [1.1200e-01, 1.7032e+01]]]),
            torch.tensor([[0.0191, 0.0189, 0.0296, 0.0150, 0.0102, 0.0146, 0.0197, 0.0184, 0.0207,
                           0.0107, 0.0126, 0.0120, 0.0232, 0.0164, 0.0152, 0.0188, 0.0127, 0.0116,
                           0.0185, 0.0291, 0.0245, 0.0277, 0.0102, 0.0249, 0.0264, 0.0286, 0.0145,
                           0.0153, 0.0191, 0.0289, 0.3064, 0.7420, 0.0987, 0.3385, 0.3346, 0.2114,
                           0.3413, 0.0943, 0.4123, 0.2794, 0.3805, 0.2506, 0.4064, 0.4569, 0.2317,
                           0.9487, 0.7856, 0.8523, 0.2755, 0.0899, 0.2570, 0.1704, 0.2007, 0.2070,
                           0.4306, 0.9110, 0.3711, 0.2128, 0.2022, 0.3078]])]
    fix = torch.tensor([[1, 4, 3, 5, 6, 2, 0]], dtype=torch.int16)
    start = time.clock()
    seq = Sequencing(seq_input=data[0])
    seq.sequncing()
    print(f"total time {time.clock() - start}")
