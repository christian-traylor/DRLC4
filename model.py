import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class QNet(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size, turn) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size ),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )
        self.target_network = None
        # self.model.apply(self.init_weights)
        self.gamma = 0.9
        self.epsilon = 1
        self.learning_rate = 1e-5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []
        self.iterations = 0
        self.sync_freq = 50
        self.turn = turn
        #self.loss_function = torch.nn.MSELoss()

    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
    # def init_weights(self, m):
        # if isinstance(m, nn.Linear):
            # nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)
