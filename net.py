import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Connect4Net(nn.Module):
#     """Neural network following AlphaGo Zero architecture with both policy and value heads"""
#     def __init__(self, device):
#         super(Connect4Net, self).__init__()
#         # Common layers - now expect 2 input channels
#         self.common = nn.Sequential(
#             #nn.Unflatten(1, (2, 6, 7)),
#             # conv 1 (input)
#             nn.Conv2d(2, 64, 4, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#             # conv 2
#             nn.Conv2d(64, 64, 4, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#             # conv 3
#             nn.Conv2d(64, 64, 4, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#         )
#
#         # Policy head
#         self.policy_head = nn.Sequential(
#             nn.Conv2d(64, 32, 1, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#             #nn.Unflatten(1, (-1, 32 * 6 * 7)),
#             nn.Linear(32 * 6 * 7, 7),
#             nn.Softmax(dim=1)
#         )
#
#         # Value head
#         self.value_head = nn.Sequential(
#             nn.Conv2d(64, 2, 1, padding=0),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#             #nn.Unflatten(1, (-1, 2 * 6 * 7)),
#             nn.Linear(2 * 6 * 7, 32),
#             nn.ReLU(),
#             #nn.Dropout(0.3),
#             nn.Linear(32, 1),
#             nn.Tanh()
#         )
#
#         self.to(device)
#
#     def forward(self, x):
#
#         common_out = self.common(x)
#         policy = self.policy_head(common_out)
#         value = self.value_head(common_out)
#
#         return policy, value


class Connect4Net(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""
    def __init__(self, device):
        super(Connect4Net, self).__init__()
        # First conv layer
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # Changed from 1 to 2 input channels
        self.bnorm1 = nn.BatchNorm2d(64)

        # Residual
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)


        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
        self.policy_lsm = nn.LogSoftmax(dim=1)

        # Value head
        self.value_conv = nn.Conv2d(64, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 6 * 7, 32)
        self.value_fc2 = nn.Linear(32, 1)

        self.to(device)

    def forward(self, x):
        # Shared layers
        x = x.view(-1, 3, 6, 7)
        x = F.relu(self.bnorm1(self.conv1(x)))

        # Residual
        res = x
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = self.bnorm3(self.conv3(x))
        x += res
        x = F.relu(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.policy_fc(policy)
        policy = self.policy_lsm(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
