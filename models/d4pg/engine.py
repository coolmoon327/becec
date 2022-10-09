import copy
from datetime import datetime
from multiprocessing import set_start_method
import torch
import torch.multiprocessing as torch_mp
import multiprocessing as mp
import numpy as np
import queue
from time import sleep
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import os

from models.agent import Agent
from utils.logger import Logger
from utils.utils import empty_torch_queue

from .d4pg import LearnerD4PG
from .networks import PolicyNetwork, ValueNetwork, AutoEncoderNetwork
from .replay_buffer import create_replay_buffer


def sampler_worker(config, replay_queue, batch_queue, replay_priorities_queue, training_on,
                   global_episode, update_step, log_dir=''):
    """
    Function that transfers replay to the buffer and batches from buffer to the queue.

    Args:
        config:
        replay_queue:
        batch_queue:
        training_on:
        global_episode:
        log_dir:
    """
    batch_size = config['batch_size']
    logger = Logger(f"{log_dir}/data_struct")

    beta = config["priority_beta_start"]

    # Create replay buffer
    replay_buffer = create_replay_buffer(config)

    while training_on.value:
        # (1) Transfer replays to global buffer
        n = replay_queue.qsize()
        for _ in range(n):
            replay = replay_queue.get()
            replay_buffer.add(*replay)

        # (2) Transfer batch of replay from buffer to the batch_queue
        if len(replay_buffer) < batch_size * 100:
            continue

        try:
            inds, weights = replay_priorities_queue.get_nowait()
            replay_buffer.update_priorities(inds, weights)
        except queue.Empty:
            pass

        try:
            batch = replay_buffer.sample(batch_size, beta)
            batch_queue.put_nowait(batch)
        except:
            # sleep(0.1)
            continue

        beta = min(config["priority_beta_end"], beta + global_episode.value/1e5)

        # Log data structures sizes
        step = update_step.value
        logger.scalar_summary("data_struct/global_episode", global_episode.value, step)
        logger.scalar_summary("data_struct/replay_queue", replay_queue.qsize(), step)
        logger.scalar_summary("data_struct/batch_queue", batch_queue.qsize(), step)
        logger.scalar_summary("data_struct/replay_buffer", len(replay_buffer), step)

    if config['save_buffer_on_disk']:
        replay_buffer.dump(config["results_path"])

    empty_torch_queue(batch_queue)
    print("Stop sampler worker.")


def learner_worker(config, training_on, policy, target_policy_net, value_net, target_value_net, learner_w_queue, replay_priority_queue,
                   batch_queue, update_step, experiment_dir):
    learner = LearnerD4PG(config, policy, target_policy_net, value_net, target_value_net, learner_w_queue, log_dir=experiment_dir)
    learner.run(training_on, batch_queue, replay_priority_queue, update_step)


def agent_worker(config, policy, value_net, learner_w_queue, global_episode, i, agent_type,
                 experiment_dir, training_on, replay_queue, update_step):
    agent = Agent(config,
                  policy=policy,
                  global_episode=global_episode,
                  n_agent=i,
                  agent_type=agent_type,
                  log_dir=experiment_dir)
    agent.set_value_net(value_net)
    agent.run(training_on, replay_queue, learner_w_queue, update_step)


class Engine(object):
    def __init__(self, config):
        self.config = config

    def train_rl(self):
        config = self.config

        batch_queue_size = config['batch_queue_size']
        n_agents = config['num_agents']

        # Create directory for experiment
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Data structures
        processes = []
        replay_queue = mp.Queue(maxsize=config['replay_queue_size'])
        training_on = mp.Value('i', 1)
        update_step = mp.Value('i', 0)
        global_episode = mp.Value('i', 0)
        learner_w_queue = torch_mp.Queue(maxsize=n_agents)
        replay_priorities_queue = mp.Queue(maxsize=config['replay_queue_size'])

        # Data sampler
        batch_queue = mp.Queue(maxsize=batch_queue_size)
        p = torch_mp.Process(target=sampler_worker,
                             args=(config, replay_queue, batch_queue, replay_priorities_queue, training_on,
                                   global_episode, update_step, experiment_dir))
        processes.append(p)

        err = False
        if config['load_param_while_training']:
            try:
                target_policy_net = torch.load(f"./results/Actor_Network_Params/M_{self.config['M']}_T_{self.config['T']}_Dt_{self.config['delta_t']}_Gamma_{self.config['discount_rate']}.pt")
                target_value_net = torch.load(f"./results/Critic_Network_Params/M_{self.config['M']}_T_{self.config['T']}_Dt_{self.config['delta_t']}_Gamma_{self.config['discount_rate']}.pt")
            except:
                err = True
        if (not config['load_param_while_training']) or err:
            # actor
            if config['env'] == 'BECEC' and config['action_mode'] == 1:
                target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], hidden_layer_num=config['hidden_layer_num'], device=config['device'], group_num=config['n_tasks'], discrete_action=True)
            else:
                target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], hidden_layer_num=config['hidden_layer_num'], device=config['device'])
            # critic
            target_value_net = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], config['v_min'], config['v_max'], config['num_atoms'], hidden_layer_num=config['hidden_layer_num'], device=config['device'])
        
        # 因为鼓励探索, 所以 Exploration 使用的 net_cpu 并不需要 load, 而是通过软更新慢慢 update
        if config['env'] == 'BECEC' and config['action_mode'] == 1:
            policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], hidden_layer_num=config['hidden_layer_num'], device=config['agent_device'], group_num=config['n_tasks'], discrete_action=True)
        else:
            policy_net_cpu = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], hidden_layer_num=config['hidden_layer_num'], device=config['agent_device'])
        # 仅在 wolp 模式下需要使用 value_net_cpu
        value_net_cpu = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], config['v_min'], config['v_max'], config['num_atoms'], hidden_layer_num=config['hidden_layer_num'], device=config['agent_device'])

        policy_net = copy.deepcopy(target_policy_net)
        value_net = copy.deepcopy(target_value_net)
        target_policy_net.share_memory()
        target_value_net.share_memory()

        # Learner (neural net training process)
        p = torch_mp.Process(target=learner_worker, args=(config, training_on, policy_net, target_policy_net, value_net, target_value_net, learner_w_queue,
                                                          replay_priorities_queue, batch_queue, update_step, experiment_dir))
        processes.append(p)

        # Single agent for exploitation
        p = torch_mp.Process(target=agent_worker,
                             args=(config, target_policy_net, target_value_net, None, global_episode, 0, "exploitation", experiment_dir,
                                   training_on, replay_queue, update_step))
        processes.append(p)

        # Agents (exploration processes)
        for i in range(1, n_agents+1):
            p = torch_mp.Process(target=agent_worker,
                                 args=(config, copy.deepcopy(policy_net_cpu), copy.deepcopy(value_net_cpu), learner_w_queue, global_episode, 
                                       i, "exploration", experiment_dir, training_on, replay_queue, update_step))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def train_encoder(self):
        config = self.config
        logger = Logger(f"./results/Auto_Encoder/encoder_training")
        net_path = f"./results/Auto_Encoder/encoder_dt_{config['delta_t']}.npy"
        
        try:
            ae_net = torch.load(net_path)
            print("Load encoder net successfully.")
        except:
            ae_net = AutoEncoderNetwork(num_input=config['delta_t'], num_output=config['ae_output_size'], hidden_size=config['ae_dense_size'], hidden_layer_num=config['ae_hidden_layer_num'], device=config['device'])

        try:
            data = np.load(f"./results/Auto_Encoder/database_dt_{config['delta_t']}.npy")
            print("Load training data successfully.")
        except:
            from env.becec.Test_Scheduler import Scheduler as Test_Scheduler
            test_scheduler = Test_Scheduler(config, 10)
            test_scheduler.run()
            data = np.array(test_scheduler.bs_observation_buffer)

        data = torch.from_numpy(data).to(config['device']).float()
        
        import torch.optim as optim
        import torch.utils.data as Data
        optimizer = optim.Adam(ae_net.parameters(), lr=config['ae_learning_rate'])
        loss_func = torch.nn.MSELoss()

        torch_dataset = Data.TensorDataset(data, data)  # auto-encoder's label is data itself
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=config['ae_batch_size'], shuffle=True)


        log_times = 0
        for epoch in range(config['ae_epoch']):
            log = []
            for step, (batch_x, batch_y) in enumerate(loader):
                output = ae_net(batch_x)
                loss = loss_func(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step%1000 == 0:
                    logger.scalar_summary("auto_encoder/loss", loss.cpu().detach().numpy(), log_times)
                    print(f"Epoch: {epoch}, step: {step}, Auto-Encoder Loss: {loss}")
                    log_times += 1
        
        torch.save(ae_net, net_path)

    def train(self, mode=0):
        """
        mode 0: train both
        mode 1: only train encoder
        mode 2: only train rl
        """
        if mode != 2:
            self.train_encoder()
            # The trained encoder will be saved locally, and RL can directly load it from the local
        if mode != 1:
            self.train_rl()

        print("End.")
    
    def test(self):
        config = self.config

        # Create directory for experiment
        experiment_dir = f"{config['results_path']}/{config['env']}-{config['model']}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        # Data structures
        replay_queue = mp.Queue(maxsize=config['replay_queue_size'])
        training_on = mp.Value('i', 2)  # Make the changes at agent.py: 0/1 used for training mode; 2 used for testing mode
        update_step = mp.Value('i', 0)
        global_episode = mp.Value('i', 0)

        try:
            target_policy_net = torch.load(f"./results/Actor_Network_Params/M_{self.config['M']}_T_{self.config['T']}_Dt_{self.config['delta_t']}_Gamma_{self.config['discount_rate']}.pt")
        except:
            print("No policy model!")
            target_policy_net = PolicyNetwork(config['state_dim'], config['action_dim'], config['dense_size'], hidden_layer_num=config['hidden_layer_num'], device=config['device'])

        try:
            target_value_net = torch.load(f"./results/Critic_Network_Params/M_{self.config['M']}_T_{self.config['T']}_Dt_{self.config['delta_t']}_Gamma_{self.config['discount_rate']}.pt")
        except:
            print("No value model!")
            target_value_net = ValueNetwork(config['state_dim'], config['action_dim'], config['dense_size'], config['v_min'], config['v_max'], config['num_atoms'], hidden_layer_num=config['hidden_layer_num'], device=config['device'])

        target_policy_net.eval()
        target_value_net.eval()

        # Single agent for exploitation
        agent_worker(config, target_policy_net, target_value_net, None, global_episode, 0, "exploitation", experiment_dir,
                                   training_on, replay_queue, update_step)

        print("End.")
