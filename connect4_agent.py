import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from board import Board, PlayerTracker


class Connect4Net(nn.Module):
    """Neural network following AlphaGo Zero architecture with both policy and value heads"""
    
    def __init__(self):
        super(Connect4Net, self).__init__()
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


class Node:
    """MCTS Node containing Q-values, prior probabilities, and visit counts"""
    
    def __init__(self, state: Board, prior=0, move_count=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = Board()
        self.state.board = state.board.copy()
        self.is_terminal = False
        self.terminal_value = None
        self.move_count = move_count  # Track move count for value decay
    
    def has_children(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Connect4Agent:
    def __init__(self, num_simulations=100, c_puct=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Connect4Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5,
                                                                    verbose=True)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def get_state_tensor(self, board: Board, current_player: PlayerTracker):
        """Convert board to tensor state with player information
        Returns a tensor with 2 channels:
        - Channel 1: The board state (1 for current player's pieces, -1 for opponent's pieces)
        - Channel 2: A plane filled with 1s (indicating it's the current player's turn)
        """
        # Convert board to current player's perspective
        board_tensor = torch.FloatTensor(board.board).unsqueeze(0)
        # Replace opponent's pieces with -1
        board_tensor[board_tensor == current_player.opponent_piece()] = -1
        # Replace current player's pieces with 1
        board_tensor[board_tensor == current_player.piece()] = 1
        
        # Player plane is always 1s (indicating it's the current player's turn)
        player_plane = torch.ones_like(board_tensor)
        
        state_tensor = torch.cat([board_tensor, player_plane], dim=0).unsqueeze(0)
        return state_tensor.to(self.device)
    
    def search(self, root_state: Board, current_player: PlayerTracker):
        """Perform MCTS search starting from root state"""
        root = Node(root_state, 0, 0)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            player = current_player.copy()
            move_count = 0  # Track moves in this simulation
            
            # Selection (DFS)
            while node.has_children():
                action, node = self.select_child(node)
                search_path.append(node)
                player.switch_players()  # Switch players
                move_count += 1
                node.move_count = move_count  # Update move count for this node
            
            # Expansion and evaluation
            state = node.state
            valid_moves = state.get_valid_moves()
            
            # Initialize terminal state variables
            is_win = False
            is_loss = False
            is_draw = False
            terminal = False
            
            # First check for immediate wins
            winning_move = None
            for move in valid_moves:
                next_state = state.get_next_state(move, player)
                if next_state.has_won(player):
                    winning_move = move
                    break
            
            if winning_move is not None:
                # If there's a winning move, only expand that one with 100% probability
                policy = np.zeros(7)
                policy[winning_move] = 1.0
                value = 1.0  # Maximum value for a win
                terminal = True
                is_win = True
            else:
                # Get policy and value from neural network
                state_tensor = self.get_state_tensor(state, player)
                with torch.no_grad():
                    policy, value = self.network(state_tensor)
                    policy = policy.cpu().numpy()[0]
                    value = value.cpu().numpy()[0][0]
                
                # Mask invalid moves
                policy_mask = np.zeros(7)
                policy_mask[valid_moves] = 1
                policy = policy * policy_mask
                if np.sum(policy) > 0:
                    policy = policy / np.sum(policy)
                
                # Check if terminal state
                is_win = state.has_won(player)
                # Switch players to make sure opponent didn't win
                player.switch_players()
                is_loss = state.has_won(player)
                # Switch back to original player
                player.switch_players()
                is_draw = len(valid_moves) == 0 and not is_win and not is_loss
                terminal = is_win or is_loss or is_draw
                
                if terminal:
                    if is_win:
                        # Winning quickly is better
                        progress = move_count / 42
                        value = 1.0 - progress * 0.7
                    elif is_loss:
                        # Losing later is better
                        progress = move_count / 42
                        value = -(1.0 - progress * 0.7)
                    else:  # Draw
                        value = 0  # Draws are neutral
            
            # Expand if not terminal
            if not terminal:
                node.is_terminal = False
                for action in valid_moves:
                    next_state = state.get_next_state(action, player)
                    child = Node(next_state, prior=policy[action], move_count=move_count + 1)
                    node.children[action] = child
            else:
                node.is_terminal = True
                node.terminal_value = value
            
            # Backpropagate with temporal discount
            self.backpropagate(search_path, value, move_count, is_draw=(terminal and is_draw))
        
        return root
    
    def select_child(self, node: Node):
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
        if child.is_terminal and child.terminal_value == 0:  # Draw
            value_score = 0  # Draws are neutral
        else:
            value_score = child.value()
        return prior_score + value_score
    
    @staticmethod
    def backpropagate(search_path, value, total_moves, is_draw=False):
        """Backpropagate value through search path with temporal discounting"""
        discount_factor = 0.95
        for node in reversed(search_path):
            # Apply temporal discount based on distance from end
            moves_to_end = total_moves - node.move_count
            discounted_value = value * (discount_factor ** moves_to_end)
            
            if is_draw:
                # For draws, always add 0
                node.value_sum += 0
            else:
                # For wins/losses, use the discounted value
                node.value_sum += discounted_value
            
                # Alternate value for next node (not needed for draws)
                value = -value
            
            node.visit_count += 1
    
    def get_action_probs(self, state: Board, current_player: PlayerTracker, temperature=1.0):
        """Get action probabilities after MCTS search"""
        # Duplicate current player information so that agent can explore without altering true current state
        agent_player = current_player.copy()
        
        root = self.search(state, agent_player)
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
        