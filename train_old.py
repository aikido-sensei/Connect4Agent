import numpy as np
import torch
from connect4_agent import Connect4Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from collections import deque
import random
 
def create_board():
     return np.zeros((6, 7))
 
def drop_piece(board, row, col, piece):
     board[row][col] = piece
 
def is_valid_location(board, col):
     return board[0][col] == 0
 
def get_next_open_row(board, col):
     for r in range(5, -1, -1):
         if board[r][col] == 0:
             return r
     return -1
 
def winning_move(board, piece):
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
 
def is_draw(board):
     return len([col for col in range(7) if board[0][col] == 0]) == 0
 
def play_self_play_game(agent, temperature=1.0):
     """Play a self-play game and return game history"""
     board = create_board()
     game_history = []
     current_player = 1
     
     while True:
         # Get action probabilities from MCTS
         action_probs = agent.get_action_probs(board, current_player, temperature)
         
         # Store state and probabilities
         game_history.append({
             'state': board.copy(),
             'current_player': current_player,
             'policy': action_probs
         })
         
         # Select action
         if temperature == 0:
             action = np.argmax(action_probs)
         else:
             # Add small noise to probabilities for exploration
             noise = np.random.dirichlet([0.3] * len(action_probs))
             action_probs = 0.75 * action_probs + 0.25 * noise
             action_probs = action_probs / np.sum(action_probs)
             action = np.random.choice(7, p=action_probs)
         
         # Make move
         row = get_next_open_row(board, action)
         if row == -1:  # Invalid move
             # Set draw value for invalid move
             for history in game_history:
                 history['value'] = 0.0
             return game_history
             
         drop_piece(board, row, action, current_player)
         
         # Check if game is over
         game_over = False
         value = 0
         
         if agent.winning_move(board, current_player):
             game_over = True
             value = 1.0
         elif is_draw(board):
             game_over = True
             value = 0.0
             
         if game_over:
             # Add final game value to all states
             for history in game_history:
                 # Value is from the perspective of the player at that state
                 if history['current_player'] == current_player:
                     history['value'] = value
                 else:
                     history['value'] = -value
             return game_history
         
         current_player = 3 - current_player  # Switch players (1 -> 2 or 2 -> 1)
 
def train_network(agent, game_histories, batch_size=32):
    """Train neural network on collected game data"""
    # Flatten game histories
    training_data = []
    for game in game_histories:
        for history in game:
            training_data.append(history)

    # Skip if not enough data
    if len(training_data) < batch_size:
        return None

    # Sample batch
    batch = random.sample(training_data, batch_size)

    # Construct 2-channel state tensors
    states = []
    for history in batch:
        board = history['state']  # shape (6, 7)
        player = history['current_player']
        
        # Convert board to current player's perspective: 1 (self), -1 (opponent), 0 (empty)
        board_input = board.copy()
        board_input[board_input == (3 - player)] = -1  # Opponent = -1
        board_input[board_input == player] = 1         # Self = 1
        board_input[board_input == 0] = 0              # Empty = 0

        player_plane = np.ones_like(board_input)       # Plane of 1s to indicate current player

        state_tensor = np.stack([board_input, player_plane])  # shape (2, 6, 7)
        states.append(state_tensor)

    states = np.array(states)  # shape (batch_size, 2, 6, 7)
    policies = np.array([history['policy'] for history in batch])
    values = np.array([history['value'] for history in batch])

    return agent.train(states, policies, values)

 
def train_agent(num_iterations=100, num_episodes=100, num_epochs=10, batch_size=32, temperature=1.0):
     """Main training loop following AlphaGo Zero methodology"""
     current_agent = Connect4Agent(num_simulations=25)  # 25 simulations is more appropriate for Connect 4
     
     # Pool of previous best models (max size 5)
     model_pool = []
     pool_size = 5
     
     # Training metrics
     metrics = {
         'policy_loss': [],
         'value_loss': [],
         'total_loss': [],
         'episode_lengths': [],
         'win_rates': [],
         'draw_rates': [],
         'loss_rates': []
     }
     
     # Create directory for saving models
     os.makedirs('models', exist_ok=True)
     
     # Keep track of best loss
     best_loss = float('inf')
     
     # Training iterations
     for iteration in range(num_iterations):
         print(f"\n{'='*50}")
         print(f"Iteration {iteration + 1}/{num_iterations}")
         print(f"{'='*50}")
         
         # Collect self-play games
         game_histories = []
         episode_lengths = []
         wins = 0
         losses = 0
         draws = 0
         
         for episode in tqdm(range(num_episodes), desc="Self-play games"):
             # More gradual temperature annealing
             progress = episode / num_episodes
             if progress < 0.9:  # Keep high temperature for 90% of episodes
                 temp = temperature
             else:
                 # Gradually decrease temperature in last 10% of episodes
                 temp = temperature * (1 - (progress - 0.9) / 0.1)
                 temp = max(0.1, temp)  # Don't go below 0.1
             
             # 50% chance to play against a previous model if available
             if model_pool and random.random() < 0.5:
                 # Create opponent agent and load random previous model
                 opponent = Connect4Agent(num_simulations=25)  # Same number of simulations for opponent
                 opponent_state = random.choice(model_pool)
                 opponent.load_state_dict(opponent_state)
                 game_history = play_game(current_agent, opponent, temp)
             else:
                 game_history = play_self_play_game(current_agent, temp)
                 
             game_histories.append(game_history)
             episode_lengths.append(len(game_history))
             
             # Count game outcomes
             final_state = game_history[-1]
             if final_state['value'] == 0:
                 draws += 1
             elif final_state['value'] == 1:  # Current player won
                 if final_state['current_player'] == 1:
                     wins += 1
                 else:
                     losses += 1
             else:  # value == -1, current player lost
                 if final_state['current_player'] == 1:
                     losses += 1
                 else:
                     wins += 1
         
         # Calculate statistics
         total_games = wins + losses + draws
         win_rate = wins / total_games
         draw_rate = draws / total_games
         loss_rate = losses / total_games
         
         metrics['win_rates'].append(win_rate)
         metrics['draw_rates'].append(draw_rate)
         metrics['loss_rates'].append(loss_rate)  # Add loss rate tracking
         
         print(f"\nGame Statistics:")
         print(f"Average game length: {np.mean(episode_lengths):.1f} moves")
         print(f"Win rate: {win_rate*100:.1f}%")
         print(f"Draw rate: {draw_rate*100:.1f}%")
         print(f"Loss rate: {loss_rate*100:.1f}%")
         
         # Train network on collected games
         epoch_losses = []
         for epoch in tqdm(range(num_epochs), desc="Training epochs"):
             result = train_network(current_agent, game_histories, batch_size)
             if result is not None:
                 total_loss, policy_loss, value_loss = result
                 metrics['total_loss'].append(total_loss)
                 metrics['policy_loss'].append(policy_loss)
                 metrics['value_loss'].append(value_loss)
                 epoch_losses.append(total_loss)
         
         if epoch_losses:
             avg_loss = np.mean(epoch_losses)
             print(f"\nAverage epoch loss: {avg_loss:.4f}")
             
             # Update model pool if current agent is good enough
             if avg_loss < best_loss:
                 best_loss = avg_loss
                 # Add current model state to pool
                 model_pool.append(current_agent.state_dict())
                 # Keep only the last pool_size models
                 if len(model_pool) > pool_size:
                     model_pool.pop(0)  # Remove oldest model
                 print(f"\nModel added to pool (pool size: {len(model_pool)})")
     
     # Save only the final model
     current_agent.save_model('models/model_latest.pth')
     
     # Create one comprehensive plot at the end
     plt.figure(figsize=(15, 5))
     
     plt.subplot(1, 3, 1)
     plt.plot(metrics['total_loss'], label='Total Loss')
     plt.plot(metrics['policy_loss'], label='Policy Loss')
     plt.plot(metrics['value_loss'], label='Value Loss')
     plt.title('Training Losses')
     plt.xlabel('Training Step')
     plt.ylabel('Loss')
     plt.legend()
     
     plt.subplot(1, 3, 2)
     plt.plot(metrics['episode_lengths'])
     plt.title('Episode Lengths')
     plt.xlabel('Episode')
     plt.ylabel('Length')
     
     plt.subplot(1, 3, 3)
     plt.plot(metrics['win_rates'], label='Win Rate')
     plt.plot(metrics['draw_rates'], label='Draw Rate')
     plt.plot(metrics['loss_rates'], label='Loss Rate')
     plt.title('Game Outcomes')
     plt.xlabel('Iteration')
     plt.ylabel('Rate')
     plt.legend()
     
     plt.tight_layout()
     plt.savefig('models/training_curves.png')
     plt.close()
         
     print("\nTraining completed!")
     print("Final model saved as: models/model_latest.pth")
     print("Training curves saved as: models/training_curves.png")
 
def play_game(agent1, agent2, temperature=1.0):
     """Play a game between two agents"""
     board = create_board()
     game_history = []
     current_player = 1
     
     while True:
         # Get current state
         state = board.copy()
         
         # Get action probabilities from current player's agent
         current_agent = agent1 if current_player == 1 else agent2
         action_probs = current_agent.get_action_probs(board, current_player, temperature)
         
         # Store state and probabilities
         game_history.append({
             'state': state.copy(),
             'current_player': current_player,
             'policy': action_probs
         })
         
         # Select action
         if temperature == 0:
             action = np.argmax(action_probs)
         else:
             # Add small noise to probabilities for exploration
             noise = np.random.dirichlet([0.3] * len(action_probs))
             action_probs = 0.75 * action_probs + 0.25 * noise
             action_probs = action_probs / np.sum(action_probs)
             action = np.random.choice(7, p=action_probs)
         
         # Make move
         row = get_next_open_row(board, action)
         if row == -1:  # Invalid move
             # Set draw value for invalid move
             for history in game_history:
                 history['value'] = 0.0
             return game_history
             
         drop_piece(board, row, action, current_player)
         
         # Check if game is over
         game_over = False
         value = 0
         
         if current_agent.winning_move(board, current_player):
             game_over = True
             value = 1.0
         elif is_draw(board):
             game_over = True
             value = 0.0
             
         if game_over:
             # Add final game value to all states
             for history in game_history:
                 # Value is from the perspective of the player at that state
                 if history['current_player'] == current_player:
                     history['value'] = value
                 else:
                     history['value'] = -value
             return game_history
         
         current_player = 3 - current_player  # Switch players (1 -> 2 or 2 -> 1)
 
if __name__ == "__main__":
     # Start training with appropriate parameters for Connect 4
     train_agent(
         num_iterations=30,     # More iterations for longer learning
         num_episodes=100,      # More games per iteration
         num_epochs=15,         # Thorough training per iteration
         batch_size=64,        # Larger batch size for stable updates
         temperature=2.0        # High temperature for exploration
     ) 