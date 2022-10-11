from turtle import forward
import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

# nn.LayerNorm = LayerNorm  # use aboved DIY batch norm
nn.LayerNorm = nn.BatchNorm1d   # use torch's offical batch norm 

class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, num_states, num_actions, hidden_size, v_min, v_max,
                 num_atoms, hidden_layer_num=2, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): size of the hidden layers
            v_min (float): minimum value for critic
            v_max (float): maximum value for critic
            num_atoms (int): number of atoms in distribution
            init_w:
        """
        super(ValueNetwork, self).__init__()
        
        self.num_atoms = num_atoms

        self.linear_in = nn.Linear(num_states + num_actions, hidden_size)

        modules = []
        for i in range(hidden_layer_num):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU()
                )
            )
        self.hidden = nn.Sequential(*modules)
            
        self.linear_last = nn.Linear(hidden_size, 64)
        self.linear_out = nn.Linear(64, num_atoms)

        if num_atoms > 1:
            self.z_atoms = np.linspace(v_min, v_max, num_atoms)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = torch.relu(self.linear_in(x))
        x = self.hidden(x)

        x = self.linear_last(x)
        x = torch.relu(self.linear_out(x))
        if self.num_atoms > 1:
            x = torch.softmax(x, dim=1)
        return x

    def to(self, device):
        super(ValueNetwork, self).to(device)
        self.device = device

    def get_probs(self, state, action):
        return self.forward(state, action)


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, hidden_layer_num=2, device='cuda', group_num=1, discrete_action=False):
        """
        Args:
            num_states (int): state dimension
            num_actions (int):  action dimension
            hidden_size (int): size of the hidden layer
            group_num (int): if use discrete actions, input the actions number, each action is an indepedent group (num_actions is splited into several groups)
                              default 1 means there is only one group, group_size = num_actions
        """
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.linear_in = nn.Linear(num_states, hidden_size)

        modules = []
        for i in range(hidden_layer_num):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU()
                )
            )
        self.hidden = nn.Sequential(*modules)

        self.linear_out = nn.Linear(hidden_size, num_actions)
        
        self.discrete_action = discrete_action
        self.groups_num = group_num
        self.softsign = nn.Softsign()
        self.softmax = nn.Softmax(dim=-1)

        self.to(device)

    def forward(self, state):
        x = torch.relu(self.linear_in(state))
        x = self.hidden(x)
        
        x = self.linear_out(x)

        if self.discrete_action:
            x = torch.chunk(x, self.groups_num, dim=-1)
            x = [self.softmax(x[i]) for i in range(self.groups_num)]
            x = torch.cat(x, dim=-1)
        else:
            x = self.softsign(x)

        return x

    def get_action(self, state):
        # actions are mapped into [-1., 1.]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action

    def to(self, device):
        super(PolicyNetwork, self).to(device)
        self.device = device


class AutoEncoderNetwork(nn.Module):
    def __init__(self, num_input, num_output, hidden_size, hidden_layer_num=2, device='cuda'):
        super(AutoEncoderNetwork, self).__init__()
        # output 指的是 encoder 编码后的输出
        self.hidden_layer_num = hidden_layer_num
        self.device = device

        # Encoder
        self.encoder_in = nn.Linear(num_input, hidden_size)
        self.encoder_hiddens = [nn.Linear(hidden_size, hidden_size) for _ in range(self.hidden_layer_num)]
        self.encoder_out = nn.Linear(hidden_size, num_output)

        # Decoder
        self.decoder_in = nn.Linear(num_output, hidden_size)
        self.decoder_hiddens = [nn.Linear(hidden_size, hidden_size) for _ in range(self.hidden_layer_num)]
        self.decoder_out = nn.Linear(hidden_size, num_input)

        self.to(device)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        x = torch.relu(self.encoder_in(x))
        for i in range(self.hidden_layer_num):
            x = torch.relu(self.encoder_hiddens[i](x))
        x = torch.relu(self.encoder_out(x))
        return x
    
    def decode(self, x):
        x = torch.relu(self.decoder_in(x))
        for i in range(self.hidden_layer_num):
            x = torch.relu(self.decoder_hiddens[i](x))
        x = torch.relu(self.decoder_out(x))
        return x

    def to(self, device):
        super(AutoEncoderNetwork, self).to(device)
        for layer in self.encoder_hiddens:
            layer.to(device)
        for layer in self.decoder_hiddens:
            layer.to(device)
        self.device = device