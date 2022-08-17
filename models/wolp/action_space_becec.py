import numpy as np
import itertools
import pyflann

"""
    This class is specific used for BECEC environment. Wolp agent should use it instead of
    action_space.py when using BECEC. Unlike action_space.py, points are initialed near the 
    inputs when the search_point function is called. A search can be made using the search_point
    function that returns the k (given) nearest neighbors of the input point.
"""

# 修改二: 自己实现一个 k means, 不再提前生成笛卡尔积
# 遍历 round(inputs) 附近的 2^n_tasks 个点 (底数也可以是 4)
# 从中选出与 inputs 的 MSE 最小的 K 个

class Space:

    def __init__(self, config):
        self.config = config
        self.M = config['M']
        self._dimensions = config['n_tasks']
        # real action range
        self._low = np.array([0 for _ in range(self._dimensions)])
        self._high = np.array([self.M for _ in range(self._dimensions)])
        self._range = self._high - self._low    # [0, 1, ..., M]
        # raw action range (from the actor network)
        self._space_low = -1.
        self._space_high = 1.
        self._k = (self._space_high - self._space_low) / self._range
        
        self._flann = pyflann.FLANN()

    def rebuild_flann(self, input):
        """在 input 附近选择一些点作为 knn 的动作空间, 从而构建 flann
        1. 输入 real_action 而非 raw_action, 不能输入 batch
        2. 在 round(real_action) 附近搜索出邻近点
        3. 找到邻近点的笛卡尔积, 作为动作空间
        """
        dims = self._dimensions
        points_in_each_axis = 3     # total points number = points_in_each_axis ** dims
        input = np.squeeze(input)

        axis = []
        for i in range(dims):
            x_in = np.round(input[i])
            x_axis = []
            # 待选节点
            points = np.array([p for p in range(int(self._low[i]), int(self._high[i])+1)])
            dis = np.abs(points - x_in)
            for _ in range(points_in_each_axis):
                min_arg = np.argmin(dis, axis=-1)
                x_axis.append(np.squeeze(points[min_arg]).tolist())
                points = np.delete(points, min_arg, -1)
                dis = np.delete(dis, min_arg, -1)
            axis.append(x_axis)
        
        space = []
        for _ in itertools.product(*axis):
            space.append(list(_))
        space = np.array(space).astype(np.float64)

        self._index = self._flann.build_index(space, algorithm='kdtree')

        return space

    def search_point(self, point, k):
        p_raw = point
        if not isinstance(point, np.ndarray):
            p_raw = np.array([p_raw]).astype(np.float64).flatten()
        p_in = self.export_point(p_raw)

        knns = []
        knns_raw = []

        batch_size = p_in.shape[0]
        for b in range(batch_size):
            space = self.rebuild_flann(p_in[b])
            search_res, _ = self._flann.nn_index(p_in[b], k)
            search_res = np.squeeze(search_res)

            knns.append(space[search_res].tolist())
            temp_raw = []
            for p in knns[b]:
                temp_raw.append(self.import_point(p))
            knns_raw.append(temp_raw)

        if k == 1:
            p_out = [p_out]
        
        knns_raw = np.array(knns_raw)
        knns = np.array(knns)
        return knns_raw, knns   # batch_size, k, point_size
    
    def import_point(self, point):
        return self._space_low + self._k * (point - self._low)

    def export_point(self, point):
        return self._low + (point - self._space_low) / self._k


'''
    test
'''

# import yaml
# with open('config_d4pg.yml', 'r') as ymlfile:
#     config = yaml.safe_load(ymlfile)
# ds = Space(config)
# point = np.array([[0. for _ in range(config['n_tasks'])]])
# point[:, 1] = -1.
# point[:, 4] = 0.6
# print(point)
# output, output_2 = ds.search_point(point, 20)
# print(output)
# print(output_2)