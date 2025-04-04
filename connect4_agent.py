import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

class Connect4Net(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""
    def __init__(self):
        super(Connect4Net, self).__init__()
        # Common layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)  # 7 possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 6 * 7)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 6 * 7)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class Node:
    """MCTS Node containing Q-values, prior probabilities, and visit counts"""
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.is_terminal = False
        self.terminal_value = None
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class Connect4Agent:
    def __init__(self, num_simulations=100, c_puct=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Connect4Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def get_state_tensor(self, board):
        """Convert board to tensor state"""
        return torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def get_valid_moves(self, board):
        """Returns list of valid moves"""
        return [col for col in range(7) if board[0][col] == 0]
    
    def winning_move(self, board, piece):
        """Check if the given piece has won on the board"""
        # Check horizontal locations
        for c in range(4):
            for r in range(6):
                if board[r][c] == piece and board[r][c+1] == piece and \
                   board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # Check vertical locations
        for c in range(7):
            for r in range(3):
                if board[r][c] == piece and board[r+1][c] == piece and \
                   board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(4):
            for r in range(3):
                if board[r][c] == piece and board[r+1][c+1] == piece and \
                   board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(4):
            for r in range(3, 6):
                if board[r][c] == piece and board[r-1][c+1] == piece and \
                   board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        
        return False
    
    def get_next_state(self, board, action, player):
        """Returns next state after taking action"""
        next_board = board.copy()
        for row in range(5, -1, -1):
            if next_board[row][action] == 0:
                next_board[row][action] = player
                break
        return next_board
    
    def search(self, root_state, current_player):
        """Perform MCTS search starting from root state"""
        root = Node(0)
        root.state = root_state
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            player = current_player
            
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                player = 3 - player  # Switch players
            
            # Expansion and evaluation
            state = node.state
            valid_moves = self.get_valid_moves(state)
            
            # Get policy and value from neural network
            state_tensor = self.get_state_tensor(state)
            with torch.no_grad():
                policy, value = self.network(state_tensor)
                policy = policy.cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]
                
                # Flip value if it's player 2's turn
                if player != current_player:
                    value = -value
            
            # Mask invalid moves
            policy_mask = np.zeros(7)
            policy_mask[valid_moves] = 1
            policy = policy * policy_mask
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
            
            # Check if terminal state
            terminal = len(valid_moves) == 0 or self.winning_move(state, 1) or self.winning_move(state, 2)
            if terminal:
                if self.winning_move(state, current_player):
                    value = 1.0
                elif self.winning_move(state, 3 - current_player):
                    value = -1.0
                else:
                    value = 0.0  # Draw
            
            # Expand if not terminal
            if not terminal:
                node.is_terminal = False
                for action in valid_moves:
                    next_state = self.get_next_state(state, action, player)
                    node.children[action] = Node(prior=policy[action])
                    node.children[action].state = next_state
            else:
                node.is_terminal = True
                node.terminal_value = value
            
            # Backpropagate
            self.backpropagate(search_path, value)
        
        return root
    
    def select_child(self, node):
        """Select child node using PUCT algorithm"""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def ucb_score(self, parent, child):
        """Calculate UCB score using PUCT algorithm"""
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # Negative because we alternate players
        return prior_score + value_score
    
    def backpropagate(self, search_path, value):
        """Backpropagate value through search path"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Alternate value for opponent
    
    def get_action_probs(self, state, current_player, temperature=1.0):
        """Get action probabilities after MCTS search"""
        root = self.search(state, current_player)
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temperature == 0:
            action_idx = np.argmax(visit_counts)
            action_probs = np.zeros(7)
            action_probs[actions[action_idx]] = 1
            return action_probs
        
        # Apply temperature
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / np.sum(visit_count_distribution)
        
        action_probs = np.zeros(7)
        action_probs[actions] = visit_count_distribution
        return action_probs
    
    def make_move(self, board, current_player=1):
        """Interface method for the game environment"""
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return -1
        
        # Use MCTS to get action probabilities
        action_probs = self.get_action_probs(board, current_player, temperature=0)  # Use temperature=0 for actual play
        return np.argmax(action_probs)
    
    def train(self, states, policies, values):
        """Train the network on a batch of data"""
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(policies).to(self.device)
        target_values = torch.FloatTensor(values).to(self.device)
        
        # Get predictions
        policy_pred, value_pred = self.network(states)
        
        # Compute losses with L2 regularization
        policy_loss = -torch.sum(target_policies * torch.log(policy_pred + 1e-8)) / states.shape[0]
        value_loss = torch.mean((value_pred - target_values) ** 2)
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in self.network.parameters():
            l2_reg = l2_reg + torch.norm(param)
        total_loss = policy_loss + value_loss + 1e-4 * l2_reg
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step(total_loss)
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def state_dict(self):
        """Get agent's state dictionary"""
        return {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        """Load agent's state from dictionary"""
        self.network.load_state_dict(state_dict['network_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)

def train_agent(num_iterations=100, num_episodes=100, num_epochs=10, batch_size=32, temperature=1.0):
    """Main training loop following AlphaGo Zero methodology"""
    agent = Connect4Agent(num_simulations=100)  # More simulations for better move selection
    
    # Training metrics
    metrics = {
        'policy_loss': [],
        'value_loss': [],
        'total_loss': [],
        'episode_lengths': []
    }

    # Training loop
    for iteration in range(num_iterations):
        for episode in range(num_episodes):
            # Training episode
            for epoch in range(num_epochs):
                # Training epoch
                for batch in range(0, len(states), batch_size):
                    # Training batch
                    states_batch = states[batch:batch+batch_size]
                    policies_batch = policies[batch:batch+batch_size]
                    values_batch = values[batch:batch+batch_size]
                    
                    loss, policy_loss, value_loss = agent.train(states_batch, policies_batch, values_batch)
                    print(f"Iteration {iteration}, Episode {episode}, Epoch {epoch}, Batch {batch}, Loss: {loss}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")

            # Save model after each iteration
            agent.save_model(f"model_iteration_{iteration}.pth")

        # Save model after all iterations
        agent.save_model("final_model.pth") 