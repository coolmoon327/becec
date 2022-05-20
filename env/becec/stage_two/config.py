import pickle
import os
import argparse
import torch
from datetime import datetime
import sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from env.becec.Parameter import Parameter

def argparser(env):
	parser = argparse.ArgumentParser()
	# main parts
	parser.add_argument('-m', '--mode', metavar = 'M', type = str, required = True, choices = ['train', 'train_emv', 'test'], help = 'train or train_emv or test')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 512, help = 'batch size, default: 512') # batch 的大小先改小一点 64
	parser.add_argument('-t', '--city_t', metavar = 'T', type = int, default = 5, help = 'number of cities(nodes), time sequence, default: 5') # 城市编号 对应 任务编号，应该改为5（任务数量）
	parser.add_argument('-s', '--steps', metavar = 'S', type = int, default = 10000, help = 'training steps(epochs), default: 10000') # 训练次数修改为 10000
	parser.add_argument('-sl', '--slots', metavar = 'SL', type = int, default = 10, help = 'used time slots, default: 10')
	
	# details
	parser.add_argument('-e', '--embed', metavar = 'EM', type = int, default = 128, help = 'embedding size') # embedding size 应该要和 hidden size 一致，先假设不变
	parser.add_argument('-hi', '--hidden', metavar = 'HI', type = int, default = 128, help = 'hidden size')
	parser.add_argument('-c', '--clip_logits', metavar = 'C', type = int, default = 10, help = 'improve exploration; clipping logits')
	parser.add_argument('-st', '--softmax_T', metavar = 'ST', type = float, default = 1.0, help = 'might improve exploration; softmax temperature default 1.0 but 2.0, 2.2 and 1.5 might yield better results')
	parser.add_argument('-o', '--optim', metavar = 'O', type = str, default = 'Adam', help = 'torch optimizer')
	parser.add_argument('-minv', '--init_min', metavar = 'MINV', type = float, default = -0.08, help = 'initialize weight minimun value -0.08~')
	parser.add_argument('-maxv', '--init_max', metavar = 'MAXV', type = float, default = 0.08, help = 'initialize weight ~0.08 maximum value')
	parser.add_argument('-ng', '--n_glimpse', metavar = 'NG', type = int, default = 1, help = 'how many glimpse function')
	parser.add_argument('-np', '--n_process', metavar = 'NP', type = int, default = 3, help = 'how many process step in critic; at each process step, use glimpse')
	parser.add_argument('-dt', '--decode_type', metavar = 'DT', type = str, default = 'sampling', choices = ['greedy', 'sampling'], help = 'how to choose next city in actor model')
	
	# train, learning rate
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-3, help = 'initial learning rate')
	parser.add_argument('--is_lr_decay', action = 'store_false', help = 'flag learning rate scheduler default true')
	parser.add_argument('--lr_decay', metavar = 'LRD', type = float, default = 0.96, help = 'learning rate scheduler, decay by a factor of 0.96 ')
	parser.add_argument('--lr_decay_step', metavar = 'LRDS', type = int, default = 5e3, help = 'learning rate scheduler, decay every 5000 steps') #fixme: 学习率是可能需要调的参数
	
	# inference
	parser.add_argument('-ap', '--act_model_path', metavar = 'AMP', type = str, help = 'load actor model path')
	parser.add_argument('--seed', metavar = 'SEED', type = int, default = 1, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-al', '--alpha', metavar = 'ALP', type = float, default = 0.99, help = 'alpha decay in active search')
	
	# path
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger default true')
	parser.add_argument('--issaver', action = 'store_false', help = 'flag model saver default true')
	parser.add_argument('-ls', '--log_step', metavar = 'LOGS', type = int, default = 10, help = 'logger timing')
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	parser.add_argument('-md', '--model_dir', metavar = 'MD', type = str, default = './Pt/', help = 'model save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	
	# GPU
	parser.add_argument('-cd', '--cuda_dv', metavar = 'CD', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE, default single GPU')
	# para = Parameter(M=50, T=100)
	args = parser.parse_args(['-m=test', '-t=10', f'-sl={env.param.delta_t}', f'-s={1}',  # f'-s={env.param.frame}
                           '-b=1', '-ap=Pt/train10_0414_14_54_step1179000_act.pt', 
                           f'--seed={env.param.seed}', '-ls=10']) # 传入参数的位置
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')
		self.task = '%s%d'%(self.mode, self.city_t)
		self.pkl_path = self.pkl_dir + '%s.pkl'%(self.task)
		self.n_samples = self.batch * self.steps
		for x in [self.log_dir, self.model_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)

def print_cfg(cfg):
	print(''.join('%s: %s\n'%item for item in vars(cfg).items()))

def dump_pkl(args, verbose = True, override = None):
	cfg = Config(**vars(args))
	if os.path.exists(cfg.pkl_path):
		# override = input(f'found the same name pkl file "{cfg.pkl_path}".\noverride this file? [y/n]:')
		override = 'y'
	with open(cfg.pkl_path, 'wb') as f:
		if override == 'n':
			raise RuntimeError('modify cfg.pkl_path in config.py as you like')			
		pickle.dump(cfg, f)
		print('--- save pickle file in %s ---\n'%cfg.pkl_path)
		if verbose:
			print_cfg(cfg)
		
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		# if verbose:
		# 	print_cfg(cfg)
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
	return cfg

def pkl_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = 'Pkl/test10.pkl', help = 'pkl file name') # train5 在 train 后面的数据表示的含义是有几个任务
	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = argparser()
	dump_pkl(args)
	# cfg = load_pkl('./Pkl/test.pkl')
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
	