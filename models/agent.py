import shutil
import os
import time
from collections import deque
from copy import deepcopy
import torch
import numpy as np

from utils.utils import OUNoise, make_gif, empty_torch_queue
from utils.logger import Logger
from env.utils import create_env_wrapper
from models.wolp.wolp_agent import WolpertingerAgent

class Agent(object):

    def __init__(self, config, policy, global_episode, n_agent=0, agent_type='exploration', log_dir=''):
        print(f"Initializing agent {n_agent}...")
        self.config = config
        self.n_agent = n_agent
        self.agent_type = agent_type
        self.max_steps = config['max_ep_length']
        self.num_episode_save = config['num_episode_save']
        self.global_episode = global_episode
        self.local_episode = 0
        self.log_dir = log_dir

        # Create environment
        self.env_wrapper = create_env_wrapper(config)
        self.ou_noise = OUNoise(dim=config["action_dim"], low=config["action_low"], high=config["action_high"], decay_period=config['decay_period'], min_sigma=config['min_sigma'], max_sigma=config['max_sigma'])
        self.ou_noise.reset()

        self.actor = policy
        print("Agent ", n_agent, self.actor.device)

        if config['wolp_mode'] > 0:
            self.wolp_agent = WolpertingerAgent(config, self.actor.device)

        # Logger
        log_path = f"{log_dir}/agent-{agent_type}-{n_agent}"
        self.logger = Logger(log_path)

    def set_value_net(self, value_net):
        """Used in Wolpertinger Architecture"""
        self.value_net = value_net
        
    def update_learner(self, learner_w_queue, training_on):
        """Update local actor to the actor from learner. """
        if not training_on.value:
            return
        try:
            if self.config['wolp_mode'] > 0:
                source_p, source_v = learner_w_queue.get_nowait()
            else:
                source_p = learner_w_queue.get_nowait()
        except:
            return
        # 更新 actor
        target_p = self.actor
        for target_param, source_param in zip(target_p.parameters(), source_p):
            w = torch.tensor(source_param).float()
            target_param.data.copy_(w)
        # 更新 critic (仅在 wolp 框架下)
        if self.config['wolp_mode'] > 0:
            target_v = self.value_net
            for target_param, source_param in zip(target_v.parameters(), source_v):
                w = torch.tensor(source_param).float()
                target_param.data.copy_(w)

    def run(self, training_on, replay_queue, learner_w_queue, update_step):
        # Initialise deque buffer to store experiences for N-step returns
        self.exp_buffer = deque()

        best_reward = -float("inf")
        rewards = []
        rewards_pure = []

        # Only used in testing mode, meaning the total number of episodes
        counts = 0 
        counts_max = self.config['test_episodes_count']

        while training_on.value and counts < counts_max:
            if training_on.value == 2:
                counts+=1

            episode_reward = 0.
            rewards.clear()
            rewards_pure.clear()
            if self.agent_type == "exploitation" and self.config["env"] == "BECEC":
                episode_null_num = 0
                episode_thrown_num = 0
                episode_outputs_times = [0 for _ in range(self.config["M"]+1)]        # 一个 episode 中, actor 输出各个值的次数
                episode_bs_selected_times = [0 for _ in range(self.config["M"])]    # 一个 episode 中, 各个 bs 被选择的次数
                episode_reward_pure = 0.    # without penalty

            num_steps = 0
            self.local_episode += 1
            self.global_episode.value += 1
            self.exp_buffer.clear()

            if self.local_episode % 100 == 0:
                print(f"Agent: {self.n_agent}  episode {self.local_episode}")

            decay_period = self.config['decay_period']
            min_epsilon = self.config['min_sigma']
            max_epsilon = self.config['max_sigma']
            M = self.config['M']

            ep_start_time = time.time()
            state = self.env_wrapper.reset()
            self.ou_noise.reset()
            done = False
            while not done:
                action = self.actor.get_action(state)
                if self.agent_type == "exploration":
                    if self.config['noise_type'] == 0:
                        # type 0: OUNoise
                        action = self.ou_noise.get_action(action, num_steps)
                        action = action.squeeze(0)
                    elif self.config['noise_type'] == 1:
                        # type 1: Epsilon Greedy (All n_tasks choices as a group)
                        epsilon = max_epsilon - (max_epsilon - min_epsilon) * min(1.0, num_steps/decay_period)
                        if np.random.randint(0, 1000) / 1000. < epsilon:
                            # 随机生成 action
                            action = np.zeros(self.config['action_dim'])
                            for i in range(self.config['n_tasks']):
                                target_bs = np.random.randint(0, M+1)
                                action[i*(M+1) + target_bs] = 1.
                            action = torch.from_numpy(action)
                        else:
                            if not isinstance(action, np.ndarray):
                                action = action.cpu().detach().numpy()
                            action = action.squeeze(0)
                    elif self.config['noise_type'] == 2:
                        # type 2: Epsilon Greedy (Every task's choice is independently)
                        epsilon = max_epsilon - (max_epsilon - min_epsilon) * min(1.0, num_steps/decay_period)
                        if not isinstance(action, np.ndarray):
                            action = action.cpu().detach().numpy()
                        action = action.squeeze(0)
                        for i in range(self.config['n_tasks']):
                            if np.random.randint(0, 1000) / 1000. < epsilon:
                                target_bs = np.random.randint(0, M+1)
                                for j in range(M+1):
                                    if j == target_bs:
                                        action[i*(M+1) + j] = 1.
                                    else:
                                        action[i*(M+1) + j] = 0.

                else:
                    action = action.detach().cpu().numpy().flatten()
                
                if not isinstance(action, np.ndarray):
                    if torch.is_tensor(action):
                        action = action.cpu().numpy().astype(np.float64)
                    else:
                        action = np.array(action)
                    if action.ndim == 0:
                        action = np.expand_dims(action, axis=0)

                if self.config['wolp_mode'] > 0:
                    if self.config['wolp_mode'] != 3 or self.agent_type == "exploitation":
                        raw_ans, ans = self.wolp_agent.wolp_action(self.value_net, state, action)
                        action = raw_ans    # 因为 env 会进行 action 的映射, 且 action 会存入 memory, 因此选择 raw
                
                next_state, reward, done = self.env_wrapper.step(action)

                if self.agent_type == "exploitation" and self.config["env"] == "BECEC":
                    details = self.env_wrapper.get_details()
                    episode_thrown_num += details[2]
                    episode_null_num += details[3]
                    episode_reward_pure += details[5]
                    rewards_pure.append(details[5])
                    actor_outputs = details[0]
                    for output in actor_outputs:
                        episode_outputs_times[output] += 1
                    target_bs = details[1]
                    for bs in target_bs:
                        if bs == -1:
                            continue
                        episode_bs_selected_times[bs] += 1
                    
                    if num_steps % 500 == 1:
                        # print(f"---\nStep {update_step.value} Episode {self.local_episode} Exploitation:\n action={action}")
                        print(f"---\nStep {update_step.value} Episode {self.local_episode}")
                        print(f"Actor outputs: {details[0]} \n Target BS: {details[1]} \n Thrown tasks: {details[2]} | Null BSs: {details[3]} | Reward: {int(details[4]*100)/100} | Pure reward: {int(details[5]*100)/100}")
                        # if self.config['is_log_max_min_Q']:
                        #     # print(f"Global min reward: {self.config['log_min']}, global max reward: {self.config['log_max']}. Recommended v_min = {self.config['log_min']*(1+self.config['discount_rate'])}, v_max = {self.config['log_max']*(1+self.config['discount_rate'])}.")
                        #     try:
                        #         print(f"Recommended v_min = {self.config['log_v_min']}, v_max = {self.config['log_v_max']}.")
                        #     except:
                        #         pass
                        print("---")

                num_steps += 1
                if num_steps == self.max_steps:
                    done = False
                episode_reward += reward
                rewards.append(reward)

                state = self.env_wrapper.normalise_state(state)
                reward = self.env_wrapper.normalise_reward(reward)

                action = np.squeeze(action)
                if action.ndim == 0:
                    action = np.expand_dims(action, axis=0)
                self.exp_buffer.append((state, action, reward))

                # We need at least N steps in the experience buffer before we can compute Bellman
                # rewards and add an N-step experience to replay memory
                if len(self.exp_buffer) >= self.config['n_step_returns']:
                    state_0, action_0, reward_0 = self.exp_buffer.popleft()
                    discounted_reward = reward_0
                    gamma = self.config['discount_rate']
                    for (_, _, r_i) in self.exp_buffer:
                        discounted_reward += r_i * gamma
                        gamma *= self.config['discount_rate']
                    # We want to fill buffer only with form explorator
                    if self.agent_type == "exploration":
                        try:
                            replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                        except:
                            pass

                state = next_state

                if done or num_steps == self.max_steps:
                    # add rest of experiences remaining in buffer
                    while len(self.exp_buffer) != 0:
                        state_0, action_0, reward_0 = self.exp_buffer.popleft()
                        discounted_reward = reward_0
                        gamma = self.config['discount_rate']
                        for (_, _, r_i) in self.exp_buffer:
                            discounted_reward += r_i * gamma
                            gamma *= self.config['discount_rate']
                        if self.agent_type == "exploration":
                            try:
                                replay_queue.put_nowait([state_0, action_0, discounted_reward, next_state, done, gamma])
                            except:
                               pass
                    break

            # Log metrics
            if training_on.value == 2:
                # Testing mode, use episode count as step
                step = counts
            else:
                step = update_step.value
            

            self.logger.scalar_summary(f"agent_{self.agent_type}/episode_reward", episode_reward, step)
            self.logger.scalar_summary(f"agent_{self.agent_type}/episode_timing", time.time() - ep_start_time, step)
            if self.agent_type == "exploitation" and self.config["env"] == "BECEC":
                self.logger.scalar_summary(f"agent_{self.agent_type}/scheduling_errors", episode_thrown_num, step)
                self.logger.scalar_summary(f"agent_{self.agent_type}/scheduling_nulls", episode_null_num, step)
                self.logger.scalar_summary(f"agent_{self.agent_type}/episode_reward_PURE", episode_reward_pure, step)

                discounted_cumulative_reward = 0.
                gamma = self.config['discount_rate']
                rewards.reverse()   # rewards list is reversed
                for r in rewards:
                    discounted_cumulative_reward = discounted_cumulative_reward * gamma + r
                self.logger.scalar_summary(f"agent_{self.agent_type}/discounted_reward", discounted_cumulative_reward, step)
                self.logger.scalar_summary(f"agent_{self.agent_type}/mean_reward", np.mean(rewards), step)

                discounted_cumulative_reward_pure = 0.
                rewards_pure.reverse()   # rewards list is reversed
                for r in rewards_pure:
                    discounted_cumulative_reward_pure = discounted_cumulative_reward_pure * gamma + r
                self.logger.scalar_summary(f"agent_{self.agent_type}/discounted_reward_PURE", discounted_cumulative_reward_pure, step)
                self.logger.scalar_summary(f"agent_{self.agent_type}/mean_reward_PURE", np.mean(rewards_pure), step)

                dict1 = {}
                for output in range(self.config["M"]+1):
                    dict1[f"output{output}"] = episode_outputs_times[output]
                self.logger.scalars_summary(f"agent_{self.agent_type}/bs_selection_OUTPUT", dict1, step)
                
                dict2 = {}
                for bs in range(self.config["M"]):
                    dict2[f"BS{bs}"] = episode_bs_selected_times[bs]
                self.logger.scalars_summary(f"agent_{self.agent_type}/bs_selection_PRACTICE", dict2, step)
            

            if self.config["save_reward_threshold"] >= 0 and training_on.value == 1:
                # Saving agent
                reward_outperformed = episode_reward - best_reward > self.config["save_reward_threshold"]
                time_to_save = self.local_episode % self.num_episode_save == 0
                if self.agent_type == "exploitation" and (time_to_save or reward_outperformed):
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                    # self.save(f"local_episode_{self.local_episode}_reward_{best_reward:4f}")
                    self.save(f"M_{self.config['M']}_T_{self.config['T']}_Dt_{self.config['delta_t']}_Gamma_{self.config['discount_rate']}")

            if self.agent_type == "exploration" and self.local_episode % self.config['update_agent_ep'] == 0:
                self.update_learner(learner_w_queue, training_on)

        empty_torch_queue(replay_queue)
        print(f"Agent {self.n_agent} done.")

    def save(self, checkpoint_name):
        # process_dir = f"{self.log_dir}/agent_{self.n_agent}"
        process_dir = "./results/Actor_Network_Params"
        if not os.path.exists(process_dir):
            os.makedirs(process_dir)
        model_fn = f"{process_dir}/{checkpoint_name}.pt"
        torch.save(self.actor, model_fn)

    def save_replay_gif(self, output_dir_name):
        import matplotlib.pyplot as plt

        dir_name = output_dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        state = self.env_wrapper.reset()
        for step in range(self.max_steps):
            action = self.actor.get_action(state)
            action = action.cpu().detach().numpy()
            next_state, reward, done = self.env_wrapper.step(action)
            img = self.env_wrapper.render()
            plt.imsave(fname=f"{dir_name}/{step}.png", arr=img)
            state = next_state
            if done:
                break

        fn = f"{self.config['env']}-{self.config['model']}-{step}.gif"
        make_gif(dir_name, f"{self.log_dir}/{fn}")
        shutil.rmtree(dir_name, ignore_errors=False, onerror=None)
        print("fig saved to ", f"{self.log_dir}/{fn}")
