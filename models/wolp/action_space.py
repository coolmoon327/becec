#!/usr/bin/env python
# -*- coding: utf-8 -*-

# [reference] Use and modified code in https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces

import numpy as np
import itertools
import pyflann

"""
    This class represents a n-dimensional unit cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
"""


class Space:

    def __init__(self, low, high, points):

        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)
        self._space_low = -1
        self._space_high = 1
        self._k = (self._space_high - self._space_low) / self._range
        
        # 设置行为空间, 并在 flann 中进行绑定
        # 这里的行为空间就是简单的在 _space_low 与 _space_high 之间生成了 points 个点
        # 此时的 space 对应神经网络在 [-1., 1.] 的输出, 自己使用时需要修改整体逻辑
        self.__space = init_uniform_space([self._space_low] * self._dimensions,
                                          [self._space_high] * self._dimensions,
                                          points)
        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def rebuild_flann(self):
        self._index = self._flann.build_index(self.__space, algorithm='kdtree')

    def search_point(self, point, k):
        p_in = point
        if not isinstance(point, np.ndarray):
            p_in = np.array([p_in]).astype(np.float64)
        # p_in = self.import_point(point)
        search_res, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[search_res]
        p_out = []
        for p in knns:
            p_out.append(self.export_point(p))

        if k == 1:
            p_out = [p_out]
        
        # 神经网络的输出在 [-1., 1.], 因此 knns 直接对应神经网络的输出, p_out 则将输出映射回了 action 空间
        return knns, np.array(p_out)

    def import_point(self, point):
        return self._space_low + self._k * (point - self._low)

    def export_point(self, point):
        return self._low + (point - self._space_low) / self._k

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]


class Discrete_space(Space):
    """
        Discrete action space with n actions (the integers in the range [0, n))
        1, 2, ..., n-1, n

        In gym: 'Discrete' object has no attribute 'high'
    """

    def __init__(self, n):  # n: the number of the discrete actions
        super().__init__([0], [n-1], n)

    def export_point(self, point):
        return np.round(super().export_point(point)).astype(int)


def init_uniform_space(low, high, points):
    dims = len(low)
    # In Discrete situation, the action space is an one dimensional space, i.e., one row
    # 因为后面要获取笛卡尔积, 所以总元素个数是列元素的 dims 次方
    points_in_each_axis = round(points**(1 / dims))

    # 得到 dims 行, 每行是 points_in_each_axis 个从 low 到 high 均匀切割的数
    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    
    # 得到笛卡尔积, 作为动作空间
    # 从每个 list 中取出一个元素, 合成新的 list: product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    # space: e.g., [[1], [2], ... ,[n-1]]
    return np.array(space)

    # 修改一: 这里直接修改为 n_tasks 个输出, M+1 种选择
    # dims = config['n_tasks']
    # points_in_each_axis = config['M'] + 1
    # axis = [[i for i in range(points_in_each_axis)] for _ in range(dims)]
    # 问题: 笛卡尔积太大!
    
    # 修改二: 自己实现一个 k means, 不再提前生成笛卡尔积
    # 遍历 round(inputs) 附近的 2^n_tasks 个点 (底数也可以是 4)
    # 从中选出与 inputs 的 MSE 最小的 K 个

'''
    test
'''
#
# ds = Space([-2], [2], 200000)
# output, output_2 = ds.search_point(1.4123456765373, 4)
# print(output_2)
# print(output_2.shape)
