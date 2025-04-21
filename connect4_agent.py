import torch
from board import *
from net import Connect4Net
from monte_carlo import Node


class Connect4Agent:
    def __init__(self, num_simulations=100, c_puct=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Connect4Net(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def get_state_tensor(self, board, current_player):
        """Convert board to tensor state with player information
        Returns a tensor with 2 channels:
        - Channel 1: The board state (1 for current player's pieces, -1 for opponent's pieces)
        - Channel 2: A plane filled with 1s (indicating it's the current player's turn)
        """
        # Make board with current player's pieces
        current_player_tensor = torch.zeros([6, 7], dtype=torch.float32)
        current_player_tensor[board == current_player] = 1

        # Make board with opponent player's pieces
        opp_player_tensor = torch.zeros([6, 7], dtype=torch.float32)
        opp_player_tensor[board == (3 - current_player)] = 1

        if current_player == 1:
            fill = 1
        else:
            fill = 0
        player_tensor = torch.full([6, 7], fill, dtype=torch.float32)

        state_tensor = torch.stack((current_player_tensor, opp_player_tensor, player_tensor), dim=-1)
        return state_tensor.to(self.device)
    
    def search(self, root_state, current_player):
        """Perform MCTS search starting from root state"""
        root = Node(current_player, root_state.copy(), 0, prior=0)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            player = current_player
            move_count = 0  # Track moves in this simulation
            
            # Selection
            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)
                player = 3 - player  # Switch players
                move_count += 1
                node.move_count = move_count  # Update move count for this node
            
            # Expansion and evaluation
            state = node.state
            valid_moves = get_valid_moves(state)
            
            # Initialize terminal state variables
            is_win = False
            is_loss = False
            is_draw = False
            terminal = False
            
            # First check for immediate wins
            w_move = None
            for move in valid_moves:
                next_state = get_next_state(state, player, move)
                if winning_move(next_state, player):
                    w_move = move
                    break
            
            if w_move is not None:
                # If there's a winning move, only expand that one with 100% probability
                policy = np.zeros(7)
                policy[w_move] = 1.0
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
                is_win = winning_move(state, player)
                is_loss = winning_move(state, 3 - player)
                is_draw = len(valid_moves) == 0 and not is_win and not is_loss
                terminal = is_win or is_loss or is_draw
                
                if terminal:
                    if is_win:
                        value = discount_value(1, move_count)
                    elif is_loss:
                        value = -discount_value(1, move_count)
                    else:  # Draw
                        value = 0  # Draws are neutral
            
            # Expand if not terminal
            if not terminal:
                node.is_terminal = False
                for action in valid_moves:
                    next_state = get_next_state(state, action, player)
                    child = Node(player, next_state, move_count + 1, prior=policy[action])
                    node.children[action] = child
            else:
                node.is_terminal = True
                node.terminal_value = value
            
            # Backpropagate with temporal discount
            Node.backpropagate(search_path, value, move_count, player, is_draw=(terminal and is_draw))
        
        return root
    
    def select_child(self, node):
        """Select child node using PUCT algorithm"""
        return node.get_best_child(self.c_puct)

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
    
    def train(self, states, policies, values, batch):
        """Train the network on a batch of data"""
        states = torch.FloatTensor(states).to(self.device)
        target_policies = torch.FloatTensor(policies).to(self.device)
        target_values = torch.FloatTensor(values).to(self.device)

        # Get predictions
        policy_pred, value_pred = self.network(states)
        policy_pred = torch.detach(policy_pred)
        value_pred = torch.detach(value_pred)
        # Compute losses with L2 regularization
        policy_loss = -torch.sum(target_policies * torch.log(policy_pred + 1e-8)) / batch

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
