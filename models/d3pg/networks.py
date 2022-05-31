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

    def __init__(self, num_states, num_actions, hidden_size, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int): action dimension
            hidden_size (int): number of neurons in hidden layers
            init_w:
        """
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_states + num_actions, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, 1)

        self.to(device)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        x = self.linear1(x)
        x = torch.relu(self.ln1(x))

        x = self.linear2(x)
        x = torch.relu(self.ln2(x))

        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """Actor - return action value given states. """

    def __init__(self, num_states, num_actions, hidden_size, device='cuda'):
        """
        Args:
            num_states (int): state dimension
            num_actions (int):  action dimension
            hidden_size (int): size of the hidden layer
            init_w:
        """
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(num_states, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, num_actions)
        # self.ln3 = nn.LayerNorm(num_actions)

        self.to(device)

    def forward(self, state):
        x = self.linear1(state)
        x = torch.relu(self.ln1(x))

        x = self.linear2(x)
        x = torch.relu(self.ln2(x))

        x = self.linear3(x)
        # x = self.ln3(x)
        x = torch.tanh(x)

        return x

    def to(self, device):
        super(PolicyNetwork, self).to(device)
        self.device = device

    def get_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action