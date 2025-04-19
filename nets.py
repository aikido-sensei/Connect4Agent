import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Net():
    def forward(self, x):
        pass


class Net3x3(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""

    def __init__(self):
        super(Net3x3, self).__init__()
        # Common layers - now expect 2 input channels
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)  # Changed from 1 to 2 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout layers
        self.dropout = nn.Dropout(0.3)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)

        # Value head
        self.value_conv = nn.Conv2d(64, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.dropout(policy)
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = self.dropout(value)
        value = value.view(-1, 32 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class Net4x4(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""

    def __init__(self):
        super(Net4x4, self).__init__()
        # Common layers - now expect 2 input channels
        self.conv1 = nn.Conv2d(2, 64, 4, padding=1)  # Changed from 1 to 2 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 4, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Dropout layers
        self.dropout = nn.Dropout(0.3)

        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(384, 7)

        # Value head
        self.value_conv = nn.Conv2d(64, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(384, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.dropout(policy)
        policy = policy.view(-1, 384)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = self.dropout(value)
        value = value.view(-1, 384)
        value = F.relu(self.value_fc1(value))
        value = self.dropout(value)
        value = torch.tanh(self.value_fc2(value))

        return policy, value
