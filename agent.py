# We need an agent that has two models
from model import QNet
import torch
import numpy as np
import random
import torch.nn as nn
input_size = 42
hidden_size = 150
output_size = 7
loss_fn = nn.MSELoss()
 
class Agent():
    def __init__(self) -> None:
        self.first_turn = QNet(input_size, hidden_size, output_size)
        self.second_turn = QNet(input_size, hidden_size, output_size)
        self.states = []

    def action(self, state, current_model): 
        qval = current_model(state)
        qval_ = qval.data.numpy()
        if (random.random() < current_model.epsilon):
            action = np.random.randint(0,7)
        else:
            action = np.argmax(qval_)
        return action, qval
    
    def reward_winning_move(self, current_model: QNet, current_model_predicted_qval):
        Y = 10
        X = current_model_predicted_qval
        Y = torch.Tensor([Y]).detach()
        loss = loss_fn(X,Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()

    def punish_losing_move(self, current_model: QNet, current_model_predicted_qval):
        Y = -10
        X = current_model_predicted_qval
        Y = torch.Tensor([Y]).detach()
        loss = loss_fn(X,Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()

    def get_loss(self, board_copy, reward, X, current_model: QNet):
        # r + (gamma * max(model(board_copy)))
        newQ = current_model(board_copy)
        maxQ = torch.max(newQ)
        Y = reward + (current_model.gamma * maxQ)
        Y = torch.Tensor([Y]).detach()
        loss = loss_fn(Y, X)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()
