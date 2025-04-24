from connect4_agent import Connect4Agent, RandomAgent
import torch
from net import NetWithResidual, NetWithoutResidual


class Hyperparameters:
    """Class that manages all tunable hyperparameters."""

    def __init__(self, config):
        self.config = config
        if config in [0, 2]:
            self.num_iterations = 100       # More iterations for thorough learning
            self.num_episodes = 30          # Fewer but higher quality episodes
            self.num_epochs = 8             # Fewer epochs to prevent overfitting
            self.batch_size = 32            # Keep batch size moderate
            self.temperature = 1.1          # Higher temperature for better exploration
            self.mcts_iters = 100           # MTCS depth medium
            self.use_discounting = False    # Simple penalty: 1 for win, -1 for loss, 0 for draw
        elif config == 1:
            self.num_iterations = 100       # More iterations for thorough learning
            self.num_episodes = 30          # Fewer but higher quality episodes
            self.num_epochs = 8             # Fewer epochs to prevent overfitting
            self.batch_size = 32            # Keep batch size moderate
            self.temperature = 1.1          # Higher temperature for better exploration
            self.mcts_iters = 100           # MTCS depth medium
            self.use_discounting = True     # Length-dependent penalty: it's better to win quickly and lose slowly
        elif config == 3:
            self.num_iterations = 100       # More iterations for thorough learning
            self.num_episodes = 30          # Fewer but higher quality episodes
            self.num_epochs = 8             # Fewer epochs to prevent overfitting
            self.batch_size = 32            # Keep batch size moderate
            self.temperature = 1.1          # Higher temperature for better exploration
            self.mcts_iters = 40            # MTCS depth small
            self.use_discounting = False    # Simple penalty: 1 for win, -1 for loss, 0 for draw

    def init_agent(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config in [0, 1, 3]:
            nnet = NetWithResidual(device)
            return Connect4Agent(nnet, device, self.use_discounting, num_simulations=self.mcts_iters, c_puct=2.0)
        elif self.config == 2:
            nnet = NetWithoutResidual(device)
            return Connect4Agent(nnet, device, self.use_discounting, num_simulations=self.mcts_iters, c_puct=2.0)
        elif self.config == -1:
            stub = NetWithoutResidual(device)  # This nnet won't actually be used, we create one to avoid exceptions
            return RandomAgent(stub, device)

    def load_by_config(self, agent: Connect4Agent):
        if self.config == -1:
            # Nothing to load for random agent
            return
        path = f"models/config_{self.config}/model_latest.pth"
        print(f"Loading model from {path}")

        agent.load_model(path)
