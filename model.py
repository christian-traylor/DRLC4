import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class QNet(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size,) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),

        )
        self.gamma = 0.9
        self.epsilon = 1
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.losses = []
        #self.loss_function = torch.nn.MSELoss()

    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
