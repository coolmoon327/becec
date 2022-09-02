#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import action_space
from . import action_space_becec
import numpy as np
import torch

class WolpertingerAgent():
    def __init__(self, config, device):
        self.config = config
        self.device = device    # wolp doesn't know which one is using it, so exploit or explorat should choose a target device
        if config['env'] == 'BECEC' or config['env'] == 'becec':
            self.action_space = action_space_becec.Space(config)
        else:
            self.action_space = action_space.Space(config['action_low'], config['action_high'], config['action_dim'])
        self.k_nearest_neighbors = config['k_nearest_neighbors']

    def wolp_action(self, critic, state, proto_action):
        raw_output = []
        output = []

        if not isinstance(state, np.ndarray):
            state = self.to_numpy(state)
        
        if state.ndim < 2:
            state = np.expand_dims(state, axis=0)

        batch_size = len(state)
        for b in range(batch_size):
            # get the proto_action's k nearest neighbors
            # raw_actions 对应神经网络的输出, actions 对应动作空间的值
            raw_actions, actions = self.action_space.search_point(proto_action[b], self.k_nearest_neighbors)
            
            s = np.expand_dims(state[b], axis=0)
            # if not isinstance(state, np.ndarray):
            #     state = self.to_numpy(state)
            # make all the state, action pairs for the critic
            s = np.tile(s, (self.k_nearest_neighbors, 1))   # 将 s 从 [1, n] 扩充到 [k, n]
            # state = state.reshape(len(raw_actions), raw_actions.shape[1], state.shape[1]) if self.k_nearest_neighbors > 1 else state.reshape(raw_actions.shape[0], state.shape[1])
            
            s = self.from_numpy_to_device(s)
            if isinstance(raw_actions, np.ndarray):
                raw_actions = self.from_numpy_to_device(raw_actions)


            # evaluate each pair through the critic
            if self.config['model'] == 'd4pg':
                # TODO 检查这里是否有效将分布变成了 Q 值
                target_value = critic.get_probs(s, raw_actions)
                target_value = target_value * torch.from_numpy(critic.z_atoms).float().to(self.device)
                actions_evaluation = torch.sum(target_value, dim=1)
            else:
                actions_evaluation = critic(s, raw_actions)

            # find the index of the pair with the maximum value
            actions_evaluation = self.to_numpy(actions_evaluation)
            max_index = np.argmax(actions_evaluation)   # check
            # max_index = max_index.reshape(len(max_index),)

            raw_actions = self.to_numpy(raw_actions)
            # return the best action, i.e., wolpertinger action from the full wolpertinger policy
            if self.k_nearest_neighbors > 1:
                # maxb = np.squeeze(actions[b, max_index[b], :]).tolist()
                # maxb_raw = np.squeeze(raw_actions[b, max_index[b], :]).tolist()
                raw_output.append(raw_actions[max_index])
                output.append(actions[max_index])
                # raw_output = raw_actions[:, max_index, :].reshape(len(raw_actions),actions.shape[-1])
                # output = actions[:, max_index, :].reshape(len(actions),actions.shape[-1])
            else:
                raw_output.append(raw_actions[0])
                output.append(actions[0])
        
        output = np.array(output)
        raw_output = np.array(raw_output)
        return raw_output, output   # batch_size, k, action_size
    
    def to_numpy(self, data):
        output = data
        if self.device != 'cpu':
            output = output.cpu()
        output = output.detach().numpy().astype(np.float64)
        return output
    
    def from_numpy_to_device(self, data):
        output = torch.from_numpy(data).float().to(self.device)
        return output
