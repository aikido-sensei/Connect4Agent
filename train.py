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
    current_player = random.choice([1, 2])  # Randomly choose starting player
    move_count = 0
    
    while True:
        # Get action probabilities from MCTS
        action_probs = agent.get_action_probs(board, current_player, temperature)
        move_count += 1
        
        # Store state and probabilities
        game_history.append({
            'state': board.copy(),
            'current_player': current_player,
            'policy': action_probs,
            'move_number': move_count,
            'value': 0  # Initialize value to 0
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
            # Penalize invalid moves heavily
            for history in game_history:
                history['value'] = -1.0 if history['current_player'] == current_player else 1.0
            return game_history
            
        drop_piece(board, row, action, current_player)
        
        # Check if game is over
        game_over = False
        value = 0
        
        if agent.winning_move(board, current_player):
            game_over = True
            # Winning quickly is better (max reward 1.0 for quick win, min 0.3 for slow win)
            progress = move_count / 42  # How far into the game are we
            value = 1.0 - progress * 0.7  # Decrease reward for longer games
        elif is_draw(board):
            game_over = True
            value = 0  # Draws are neutral
        
        if game_over:
            # Update values for all positions in the game
            for history in game_history:
                if history['current_player'] == current_player:
                    if value == 0:  # Draw
                        history['value'] = value  # Both players get 0 for draw
                    else:
                        history['value'] = value  # Winner gets positive value
                else:
                    if value == 0:  # Draw
                        history['value'] = value  # Both players get 0 for draw
                    else:
                        history['value'] = -value  # Loser gets negative of winner's value
            return game_history
        
        current_player = 3 - current_player  # Switch players (1 -> 2 or 2 -> 1)

def train_network(agent, game_histories, batch_size=32):
    """Train neural network on collected game data"""
    # Flatten game histories
    training_data = []
    for game in game_histories:
        # Include all positions, including draws
        for history in game:
            training_data.append(history)
    
    # Skip if not enough data
    if len(training_data) < batch_size:
        return None
    
    # Sort by absolute value to prioritize decisive positions
    training_data.sort(key=lambda x: abs(x['value']), reverse=True)
    
    # Take top 75% of positions
    training_data = training_data[:int(len(training_data) * 0.75)]
    
    # Sample batch from remaining good positions
    batch = random.sample(training_data, min(batch_size, len(training_data)))
    
    # Prepare training data with 2 channels
    states = []
    for history in batch:
        board_tensor = history['state']
        player_plane = np.full_like(board_tensor, float(history['current_player'] == 1))
        state = np.stack([board_tensor, player_plane])  # Stack along channel dimension
        states.append(state)
    
    states = np.array(states)
    policies = np.array([history['policy'] for history in batch])
    values = np.array([history['value'] for history in batch])
    
    # Train network
    return agent.train(states, policies, values)

def train_agent(num_iterations=100, num_episodes=100, num_epochs=10, batch_size=32, temperature=1.0):
    """Main training loop following AlphaGo Zero methodology"""
    current_agent = Connect4Agent(num_simulations=100, c_puct=2.0)  # More simulations and higher exploration
    
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
        'loss_rates': [],
        "draws": [],
        "losses": [],
        "wins": []
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

        if iteration % 20 == 0:
            current_agent.save_model('models/model_latest_' + str(iteration) + '.pth')
            save_metrics(metrics, iteration)
        
        # Collect self-play games
        game_histories = []
        episode_lengths = []
        wins = 0
        losses = 0
        draws = 0
        
        for episode in tqdm(range(num_episodes), desc="Self-play games"):
            # More gradual temperature annealing
            progress = episode / num_episodes
            if progress < 0.8:  # Keep high temperature for 80% of episodes
                temp = temperature
            else:
                # Gradually decrease temperature in last 20% of episodes
                temp = temperature * (1 - (progress - 0.8) / 0.2)
                temp = max(0.5, temp)  # Don't go below 0.5 to maintain some exploration
            
            # 50% chance to play against a previous model if available
            if model_pool and random.random() < 0.70:
                # Create opponent agent and load random previous model
                opponent = Connect4Agent(num_simulations=100, c_puct=2.0)  # Same settings for opponent
                opponent_state = random.choice(model_pool)
                opponent.load_state_dict(opponent_state)
                game_history = play_game(current_agent, opponent, temp)
            else:
                game_history = play_self_play_game(current_agent, temp)
                
            game_histories.append(game_history)
            episode_lengths.append(len(game_history))
            metrics["episode_lengths"].append(len(game_history))

            # Count game outcomes
            final_state = game_history[-1]
            if final_state['value'] == 0:  # Draw
                draws += 1
                metrics["draws"].append(1)
                metrics["wins"].append(0)
                metrics["losses"].append(0)
            elif final_state['value'] > 0:  # Win
                wins += 1
                metrics["draws"].append(0)
                metrics["wins"].append(1)
                metrics["losses"].append(0)
            else:  # Loss
                losses += 1
                metrics["draws"].append(0)
                metrics["wins"].append(0)
                metrics["losses"].append(1)
        
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
    save_metrics(metrics, "")
        
    print("\nTraining completed!")
    print("Final model saved as: models/model_latest.pth")
    print("Training curves saved as: models/training_curves_.png")

def play_game(agent1, agent2, temperature=1.0):
    """Play a game between two agents"""
    board = create_board()
    game_history = []
    current_player = random.choice([1, 2])  # Randomly choose starting player
    move_count = 0
    
    while True:
        # Get current state
        state = board.copy()
        move_count += 1
        
        # Get action probabilities from current player's agent
        current_agent = agent1 if current_player == 1 else agent2
        action_probs = current_agent.get_action_probs(board, current_player, temperature)
        
        # Store state and probabilities
        game_history.append({
            'state': state.copy(),
            'current_player': current_player,
            'policy': action_probs,
            'move_number': move_count,
            'value': 0  # Initialize value to 0
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
            # Penalize invalid moves heavily
            for history in game_history:
                history['value'] = -1.0 if history['current_player'] == current_player else 1.0
            return game_history
            
        drop_piece(board, row, action, current_player)
        
        # Check if game is over
        game_over = False
        value = 0
        
        if current_agent.winning_move(board, current_player):
            game_over = True
            # Winning quickly is better (max reward 1.0 for quick win, min 0.3 for slow win)
            progress = move_count / 42  # How far into the game are we
            value = 1.0 - progress * 0.7  # Decrease reward for longer games
        elif is_draw(board):
            game_over = True
            value = 0  # Draws are neutral
            
        if game_over:
            # Update values for all positions in the game
            for history in game_history:
                if history['current_player'] == current_player:
                    if value == 0:  # Draw
                        history['value'] = value  # Both players get 0 for draw
                    else:
                        history['value'] = value  # Winner gets positive value
                else:
                    if value == 0:  # Draw
                        history['value'] = value  # Both players get 0 for draw
                    else:
                        history['value'] = -value  # Loser gets negative of winner's value
            return game_history
        
        current_player = 3 - current_player  # Switch players (1 -> 2 or 2 -> 1)


def save_metrics(metrics, i):
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
    plt.savefig(f'models/training_curves_{i}.png')
    plt.close()

    with open(f'models/csv_data_{i}', 'w') as f:
        f.write("total_loss,policy_loss,value_loss,episode_lengths,wins,draws,losses\n")
        for i in range(len(metrics['total_loss'])):
            f.write(f"{metrics['total_loss'][i]},{metrics['policy_loss'][i]},{metrics['value_loss'][i]},{metrics['episode_lengths'][i]},{metrics['wins'][i]},{metrics["draws"][i]},{metrics["losses"][i]}\n")


if __name__ == "__main__":
    # Start training with improved parameters
    train_agent(
        num_iterations=5000,     # More iterations for thorough learning
        num_episodes=30,       # Fewer but higher quality episodes
        num_epochs=8,         # Fewer epochs to prevent overfitting
        batch_size=32,        # Keep batch size moderate
        temperature=1.5        # Higher temperature for better exploration
    ) 