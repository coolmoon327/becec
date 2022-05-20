import torch
import torch.nn as nn
import torch.nn.functional as F

from env.becec.stage_two.config import Config, load_pkl, pkl_parser
from env.becec.stage_two.env import Env_tsp

class Greedy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, log_p):
		return torch.argmax(log_p, dim = 1).long()

class Categorical(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, log_p):
		return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

# https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Neural%20Combinatorial%20Optimization.ipynb
class PtrNet1(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.Embedding = nn.Linear(2, cfg.embed, bias = False) #fixme: 输入维度可能就不是 2 了 
		self.Env_Embedding = nn.Linear(cfg.slots * 2, cfg.embed, bias = False) # 刚刚好环境的维度也是 2, 但是融合了slots 和 p c 数据
		self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		self.Decoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
		if torch.cuda.is_available():
			self.Vec = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
			self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
		else:
			self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
			self.Vec2 = nn.Parameter(torch.FloatTensor(cfg.embed)) # what does it mean by vec, feature vector ?
		self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
		self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
		self.W_q2 = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
		self.W_ref2 = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
		self.dec_input = nn.Parameter(torch.FloatTensor(cfg.embed))
		self._initialize_weights(cfg.init_min, cfg.init_max)
		self.clip_logits = cfg.clip_logits
		self.softmax_T = cfg.softmax_T
		self.n_glimpse = cfg.n_glimpse
		self.city_selecter = {'greedy': Greedy(), 'sampling': Categorical()}.get(cfg.decode_type, None)
	
	def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
		for param in self.parameters():
			nn.init.uniform_(param.data, init_min, init_max) # make all parameters in ptr1 uniform
		
	def forward(self, x, device):
		'''	x: (task_data, env_data)
  			task_data: (batch, city_t, w_punish)
			env_data: (batch, c_p)
			enc_h: (batch, city_t, embed)
			dec_input: (batch, 1, embed)
			h: (1, batch, embed)
			return: pi: (batch, city_t), ll: (batch)
		'''
		task_data, env_data = x
		
		env_data = env_data.to(device)
		batch, city_t, _ = task_data.size()
		batch, slots_features = env_data.size()
		task_data = task_data.to(device)
		
		embed_enc_inputs = self.Embedding(task_data)
		env_embed_enc_inputs = self.Env_Embedding(env_data) #todo: 做一次扩展，env_embed..的大小应该与data_embed..一致
		env_embed_enc_inputs = env_embed_enc_inputs.unsqueeze(1).repeat(1, city_t, 1)
		embed_enc_inputs = embed_enc_inputs.add(env_embed_enc_inputs)
		embed = embed_enc_inputs.size(2)
		mask = torch.zeros((batch, city_t), device = device)
		enc_h, (h, c) = self.Encoder(embed_enc_inputs, None) #fixme: 进入这一步之前应该要将embed_enc_inputs变为两个embedding的相加
		ref = enc_h
		pi_list, log_ps = [], [] # pi_list: pointer list ?
		dec_input = self.dec_input.unsqueeze(0).repeat(batch,1).unsqueeze(1).to(device)
		for i in range(city_t):
			_, (h, c) = self.Decoder(dec_input, (h, c))
			query = h.squeeze(0)
			for i in range(self.n_glimpse):
				query = self.glimpse(query, ref, mask)
			logits = self.pointer(query, ref, mask)	
			log_p = torch.log_softmax(logits, dim = -1) # the probability after logaithem.
			next_node = self.city_selecter(log_p)
			dec_input = torch.gather(input = embed_enc_inputs, dim = 1, index = next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed))
			
			pi_list.append(next_node)
			log_ps.append(log_p)
			mask += torch.zeros((batch,city_t), device = device).scatter_(dim = 1, index = next_node.unsqueeze(1), value = 1) # mask the visited cities
			
		pi = torch.stack(pi_list, dim = 1) # routing by pointer mechanism
		ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi) # ll: log likelihood for one batch
		return pi, ll # pi 是路径， ll 是该路径出现的可能性
	
	def glimpse(self, query, ref, mask, inf = 1e8):
		"""	-ref about torch.bmm, torch.matmul and so on
			https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
			https://qiita.com/shinochin/items/aa420e50d847453cc296
			
				Args: 
			query: the hidden state of the decoder at the current
			(batch, 128)
			ref: the set of hidden states from the encoder. 
			(batch, city_t, 128)
			mask: model only points at cities that have yet to be visited, so prevent them from being reselected
			(batch, city_t)
		"""
		u1 = self.W_q(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
		u2 = self.W_ref(ref.permute(0,2,1))# u2: (batch, 128, city_t)
		V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1) # why Vec accur data, from where
		u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
		# V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
		u = u - inf * mask # inf : whether you are concerned about mask ?
		a = F.softmax(u / self.softmax_T, dim = 1) # a contain the probability of visiting some cities.
		d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
		# u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
		return d # 返回关注每个城市的关注程度，或者说有多大的概率去访问它

	def pointer(self, query, ref, mask, inf = 1e8):
		"""	Args: 
			query: the hidden state of the decoder at the current
			(batch, 128)
			ref: the set of hidden states from the encoder. 
			(batch, city_t, 128)
			mask: model only points at cities that have yet to be visited, so prevent them from being reselected
			(batch, city_t)
		"""
		u1 = self.W_q2(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
		u2 = self.W_ref2(ref.permute(0,2,1))# u2: (batch, 128, city_t)
		V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
		u = torch.bmm(V, self.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
		# V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
		u = u - inf * mask
		return u

	def get_log_likelihood(self, _log_p, pi):
		"""	args:
			_log_p: (batch, city_t, city_t)
			pi: (batch, city_t), predicted tour
			return: (batch)
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
		return torch.sum(log_p.squeeze(-1), 1)

def random_train_env(batch, slots, seed = None):
	'''
	环境随机训练数据
	return node:(batch,slots * 2)
	'''
	if seed is not None:
		torch.manual_seed(seed)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	env_computation_power = 10 * torch.rand((batch, slots, 1), device = device) + 5
	env_price = 9 * torch.rand((batch, slots, 1), device = device) + 1 # fixme: env 参数名需要改变
	env_info = torch.cat((env_computation_power, env_price), 2)
	# reshape env to (batch, slots * 2)
	env_info = torch.reshape(env_info, (batch, slots * 2))
	return env_info # 可以模仿的产生 通过 cat 拼接 两次产生的tensor

def	test_embedding(x, device):
	"""测试 nn 的映射功能 fine

	Args:
		x (batch, slots, 2): [环境信息映射]
		return (batch, embedding_env) : 将所有slots的信息映射为一个向量，维度不匹配？
	"""    
	batch, slots_features = x.size()
	x = x.to(device) # 改变形状的尝试没有问题
	embedding_env = 128 # fixme: 这里设置为所有 embedding 的统一大小
	Embedding = nn.Linear(slots_features, embedding_env, bias = False) # 做线性映射，映射指定为最后一个维度
	Embedding = Embedding.to(device) # 模型也要放在 cuda 上
	embedded_env = Embedding(x)
	print(embedded_env.size())
    
				
if __name__ == '__main__':
	# cfg = load_pkl(pkl_parser().path)
	# model = PtrNet1(cfg)
	# inputs = torch.randn(3,20,2)	
	# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	# pi, ll = model(inputs, device = device)	
	# print('pi:', pi.size(), pi)
	# print('log_likelihood:', ll.size(), ll)

	# cnt = 0
	# for i, k in model.state_dict().items():
	# 	print(i, k.size(), torch.numel(k))
	# 	cnt += torch.numel(k)	
	# print('total parameters:', cnt)

	# # ll.mean().backward()
	# # print(model.W_q.weight.grad)

	# cfg.batch = 3
	# env = Env_tsp(cfg)
	# cost = env.stack_l(inputs, pi)
	# print('cost:', cost.size(), cost)
	# cost = env.stack_l_fast(inputs, pi)
	# print('cost:', cost.size(), cost)
 
	# test nn.Linear
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	x = random_train_env(batch=4, slots=3)
	test_embedding(x, device)
	