'''
Author: your name
Date: 2021-12-25 22:25:04
LastEditTime: 2022-03-17 10:57:27
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /TSP_DRL_PtrNet-master/search.py
'''
from math import pi
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
    greedy = Greedy(test_input,
                    torch.tensor(seq.res).unsqueeze(0))
    greedy.greed_score()

    return greedy.score, greedy.u, greedy.traceInfo


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
