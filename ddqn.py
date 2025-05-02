import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, actions, input_dim, lr):
        super(DuelingDeepQNetwork, self).__init__()
        #self.fc1 = nn.Linear(*input_dim, 512)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()

        )

        conv_output_size = self.get_conv_out(input_dim)

        self.fc_advantage = nn.Sequential(
            self.conv_layer,
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions)
        )

        self.fc_value = nn.Sequential(
            self.conv_layer,
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)

    def get_conv_out(self, shape):
        out = self.conv_layer(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(out.size()))

    def forward(self, state):
        advantage = self.fc_advantage(state)
        value = self.fc_value(state)


        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

