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

nn.LayerNorm = LayerNorm

class ValueNetwork(nn.Module):
    """Critic - return Q value from given states and actions. """

    def __init__(self, num_states, num_actions, hidden_size, v_min, v_max,
                 num_atoms, device='cuda'):
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

        self.linear_in = nn.Linear(num_states + num_actions, hidden_size)
        self.ln_in = nn.LayerNorm(hidden_size)

        self.hidden_layer_num = 4
        self.hidden_linears = [nn.Linear(hidden_size, hidden_size).to(device) for _ in range(self.hidden_layer_num)]
        self.lns = [nn.LayerNorm(hidden_size).to(device) for _ in range(self.hidden_layer_num)]

        self.linear_out = nn.Linear(hidden_size, num_atoms)

        self.z_atoms = np.linspace(v_min, v_max, num_atoms)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = self.linear_in(x)
        x = torch.relu(self.ln_in(x))

        for i in range(self.hidden_layer_num):
            x = self.hidden_linears[i](x)
            x = torch.relu(self.lns[i](x))

        x = torch.softmax(self.linear_out(x), dim=1)
        return x

    def get_probs(self, state, action):
        return self.forward(state, action)


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, device='cuda', group_num=1, discrete_action=False):
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
        self.ln_in = nn.LayerNorm(hidden_size)

        self.hidden_layer_num = 4
        self.hidden_linears = [nn.Linear(hidden_size, hidden_size).to(device) for _ in range(self.hidden_layer_num)]
        self.lns = [nn.LayerNorm(hidden_size).to(device) for _ in range(self.hidden_layer_num)]

        self.linear_out = nn.Linear(hidden_size, num_actions)
        
        self.discrete_action = discrete_action
        self.groups_num = group_num
        self.softsign = nn.Softsign()
        self.softmax = nn.Softmax(dim=-1)

        self.to(device)

    def forward(self, state):
        x = self.linear_in(state)
        x = torch.relu(self.ln_in(x))

        for i in range(self.hidden_layer_num):
            x = self.hidden_linears[i](x)
            x = torch.relu(self.lns[i](x))
        
        x = self.linear_out(x)

        if self.discrete_action:
            x = torch.chunk(x, self.groups_num, dim=-1)
            x = [self.softmax(x[i]) for i in range(self.groups_num)]
            x = torch.cat(x, dim=-1)
        else:
            x = self.softsign(x)

        return x

    def to(self, device):
        super(PolicyNetwork, self).to(device)
        self.device = device

    def get_action(self, state):
        # actions are mapped into [-1., 1.]
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action