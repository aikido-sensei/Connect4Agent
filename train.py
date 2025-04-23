import argparse

from connect4_agent import Connect4Agent
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from board import *
from monte_carlo import Node
from hyperparameters import Hyperparameters


def make_move(agent: Connect4Agent, board: np.ndarray, current_player: int, game_history: list,
              move_count: int, temperature: float):
    # Get current state
    state = board.copy()
    # Get action probabilities from MCTS
    action_probs = agent.get_action_probs(state, current_player, temperature)

    # Store state and probabilities
    game_history.append({
        'state': state.copy(),
        'current_player': current_player,
        'policy': action_probs,
        'move_number': move_count,
        'value': 0  # Initialize value to 0
    })

    # Select action
    action = Node.choose_action(action_probs, temperature)

    # Make move
    row = get_next_open_row(board, action)
    if row == -1:  # Invalid move
        # Penalize invalid moves heavily
        for history in game_history:
            history['value'] = -1.0 if history['current_player'] == current_player else 1.0
        return True

    drop_piece(board, row, action, current_player)

    # Check if game is over
    game_over = False
    value = 0

    if winning_move(board, current_player):
        game_over = True
        value = discount_value(1, move_count, agent.use_discounting)
    elif is_draw(board):
        game_over = True
        value = 0  # Draws are neutral

    if game_over:
        # Update values for all positions in the game
        for history in game_history:
            if value == 0:  # Draw
                history['value'] = 0
            elif history['current_player'] == current_player:
                history['value'] = value  # Winner gets positive value
            else:
                history['value'] = -value  # Loser gets negative of winner's value
    return game_over


def play_self_play_game(agent, temperature, starting_player):
    """Play a self-play game and return game history"""
    board = create_board()
    game_history = []
    move_count = 0
    current_player = starting_player

    while True:
        move_count += 1
        over = make_move(agent, board, current_player, game_history, move_count, temperature)
        if over:
            return game_history
        current_player = change_players(current_player)  # Switch players (1 -> 2 or 2 -> 1)


def play_game(agent1, agent2, temperature, starting_player):
    """Play a game between two agents"""
    board = create_board()
    game_history = []
    # current_player = random.choice([1, 2])  # Randomly choose starting player
    move_count = 0
    current_player = starting_player

    while True:
        move_count += 1
        current_agent = agent1 if current_player == 1 else agent2
        over = make_move(current_agent, board, current_player, game_history, move_count, temperature)
        if over:
            return game_history
        current_player = change_players(current_player)  # Switch players (1 -> 2 or 2 -> 1)


def train_network(agent, game_histories, batch_size):
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
        # player_plane = np.full_like(board_tensor, float(history['current_player'] == 1))
        # state = np.stack([board_tensor, player_plane])  # Stack along channel dimension
        states.append(agent.get_state_tensor(board_tensor, history['current_player']).cpu().numpy())

    states = np.array(states)
    policies = np.array([history['policy'] for history in batch])
    values = np.array([history['value'] for history in batch])
    values = values.reshape(-1, 1)
    # Train network
    return agent.train(states, policies, values, batch_size)


def train_agent(params: Hyperparameters):
    """Main training loop following AlphaGo Zero methodology"""
    current_agent = params.init_agent()  # initialize agent based on hyperparameters

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
    os.makedirs(f'models/config_{params.config}/', exist_ok=True)

    # Keep track of best loss
    best_loss = float('inf')

    # Training iterations
    for iteration in range(params.num_iterations):
        print(f"\n{'=' * 50}")
        print(f"Iteration {iteration + 1}/{params.num_iterations}")
        print(f"{'=' * 50}")

        if iteration % 10 == 0:
            current_agent.save_model(f'models/config_{params.config}/model_latest_' + str(iteration) + '.pth')
            save_metrics(metrics, iteration, params)

        # Collect self-play games
        game_histories = []
        episode_lengths = []
        wins = 0
        losses = 0
        draws = 0

        for episode in tqdm(range(params.num_episodes), desc="Self-play games"):
            # More gradual temperature annealing
            progress = episode / params.num_episodes
            if progress < 0.8:  # Keep high temperature for 80% of episodes
                temp = params.temperature
            else:
                # Gradually decrease temperature in last 20% of episodes
                temp = params.temperature * (1 - (progress - 0.8) / 0.2)
                temp = max(0.5, temp)  # Don't go below 0.5 to maintain some exploration

            # Randomly choose starting player. Also determines agent we care about
            current_player = random.choice([1, 2])

            # 50% chance to play against a previous model if available
            if model_pool and random.random() < 0.70:

                # Create opponent agent and load random previous model
                opponent = params.init_agent()  # Same settings for opponent
                opponent_state = random.choice(model_pool)
                opponent.load_state_dict(opponent_state)
                if current_player == 1:
                    game_history = play_game(current_agent, opponent, temp, current_player)
                else:
                    game_history = play_game(opponent, current_agent, temp, current_player)
            else:
                game_history = play_self_play_game(current_agent, temp, current_player)

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
            elif final_state['current_player'] == current_player:  # current player ended the game
                if final_state['value'] > 0:  # Win
                    wins += 1
                    metrics["draws"].append(0)
                    metrics["wins"].append(1)
                    metrics["losses"].append(0)
                else:  # Loss
                    losses += 1
                    metrics["draws"].append(0)
                    metrics["wins"].append(0)
                    metrics["losses"].append(1)
            else:                                               # opponent ended the game
                if final_state['value'] > 0:  # Win
                    losses += 1
                    metrics["draws"].append(0)
                    metrics["wins"].append(0)
                    metrics["losses"].append(1)
                else:  # Loss
                    wins += 1
                    metrics["draws"].append(0)
                    metrics["wins"].append(1)
                    metrics["losses"].append(0)

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
        print(f"Win rate: {win_rate * 100:.1f}%")
        print(f"Draw rate: {draw_rate * 100:.1f}%")
        print(f"Loss rate: {loss_rate * 100:.1f}%")

        # Train network on collected games
        epoch_losses = []
        for _ in tqdm(range(params.num_epochs), desc="Training epochs"):
            result = train_network(current_agent, game_histories, params.batch_size)
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
    current_agent.save_model(f'models/config_{params.config}/model_latest.pth')

    # Create one comprehensive plot at the end
    save_metrics(metrics, "", params)

    print("\nTraining completed!")
    print(f"Final model saved as: models/config_{params.config}/model_latest.pth")
    print(f"Training curves saved as: models/config_{params.config}/training_curves_.png")


def save_metrics(metrics, i, params:Hyperparameters):
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
    plt.savefig(f'models/config_{params.config}/training_curves_{i}.png')
    plt.close()

    with open(f'models/config_{params.config}/csv_data_{i}', 'w') as f:
        f.write("total_loss,policy_loss,value_loss,episode_lengths,wins,draws,losses\n")
        for i in range(len(metrics['total_loss'])):
            f.write(
                f"{metrics['total_loss'][i]},{metrics['policy_loss'][i]},{metrics['value_loss'][i]},"
                f"{metrics['episode_lengths'][i]},{metrics['wins'][i]},{metrics["draws"][i]},{metrics["losses"][i]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a connect-4 agent. The hyperparameters in each config are described in the paper.")
    parser.add_argument("--config", type=int, choices=[0, 1, 2, 3], default=0, help="Config to train")

    args = parser.parse_args()
    # Start training with improved parameters
    train_agent(
        Hyperparameters(args.config)
    )
