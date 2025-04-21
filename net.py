import torch
import torch.nn as nn
import torch.nn.functional as F


class NetWithResidual(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""
    def __init__(self, device):
        super(NetWithResidual, self).__init__()
        # First conv layer
        # Channel 1 for player 1 pieces, channel 2 for player 2 pieces, channel 3 for player turn
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64)

        # Residual layer
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)  # Map to 7 actions
        self.policy_lsm = nn.Softmax(dim=1)

        # Value head
        self.value_conv = nn.Conv2d(64, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 6 * 7, 32)
        self.value_fc2 = nn.Linear(32, 1)  # Map to 1 state value

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
        value = value.view(-1, 3 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class NetWithoutResidual(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""
    def __init__(self, device):
        super(NetWithoutResidual, self).__init__()
        # First conv layer
        # Channel 1 for player 1 pieces, channel 2 for player 2 pieces, channel 3 for player turn
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bnorm3 = nn.BatchNorm2d(64)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)  # Map to 7 actions
        self.policy_lsm = nn.Softmax(dim=1)

        # Value head
        self.value_conv = nn.Conv2d(64, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 6 * 7, 32)
        self.value_fc2 = nn.Linear(32, 1)  # Map to 1 state value

        self.to(device)

    def forward(self, x):
        # Shared layers
        x = x.view(-1, 3, 6, 7)
        x = F.relu(self.bnorm1(self.conv1(x)))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = F.relu(self.bnorm3(self.conv3(x)))

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.policy_fc(policy)
        policy = self.policy_lsm(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 3 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value
