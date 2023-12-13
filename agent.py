from model import QNet
import torch
import numpy as np
import random
import torch.nn as nn
from copy import deepcopy
input_size = 42
hidden_size = 150
output_size = 7
loss_fn = nn.MSELoss()
 
class Agent():
    def __init__(self) -> None:
        self.first_turn = QNet(input_size, hidden_size, output_size, 0)
        self.first_turn_target_network = self.first_turn_target_network = self.initialize_target_network(self.first_turn)
        #self.first_turn.target_network = self.initialize_target_network(self.first_turn)
        self.second_turn = QNet(input_size, hidden_size, output_size, 1)
        self.second_turn_target_network = self.second_turn_target_network = self.initialize_target_network(self.second_turn)
        #self.second_turn.target_network = self.initialize_target_network(self.second_turn)
        self.states = []
        self.target_dictionary = { 
            0 : self.first_turn_target_network, 
            1: self.second_turn_target_network
        }

    def action(self, state, current_model): 
        qval = current_model(state)
        qval_ = qval.data.numpy()
        if (random.random() < current_model.epsilon):
            action = np.random.randint(0,7)
        else:
            action = np.argmax(qval_)
        return action, qval
    
    def reward_winning_move(self, current_model: QNet, current_model_predicted_qval, turn):
        X = current_model_predicted_qval
        Y = torch.Tensor([10.]).detach()
        loss = loss_fn(X,Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()
        if current_model.iterations % current_model.sync_freq == 0:
            targ_network = self.get_target_network(current_model)
            targ_network.load_state_dict(current_model.state_dict())

    def punish_losing_move(self, current_model: QNet, current_model_predicted_qval, turn):
        X = current_model_predicted_qval
        # Y = torch.Tensor([-10.])
        # Y.requires_grad = True
        Y = torch.Tensor([-10.]).detach()
        loss = loss_fn(X,Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()
        current_model.iterations += 1
        if current_model.iterations % current_model.sync_freq == 0:
            targ_network = self.get_target_network(current_model)
            targ_network.load_state_dict(current_model.state_dict())

    def get_loss(self, board_copy, reward, X, current_model: QNet, turn):
        # r + (gamma * max(model(board_copy)))
        newQ = current_model(board_copy)
        maxQ = torch.max(newQ)
        Y = reward + (current_model.gamma * maxQ)
        Y = torch.Tensor([Y]).detach()
        loss = loss_fn(X, Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()
        current_model.iterations += 1
        if current_model.iterations % current_model.sync_freq == 0:
            targ_network = self.get_target_network(current_model)
            targ_network.load_state_dict(current_model.state_dict())

    def calculate_q(self, board_copy, reward, X, current_model: QNet, turn):
        targ_network = self.get_target_network(current_model)
        with torch.no_grad():
            Q2 = targ_network(board_copy)
        maxQ = torch.max(Q2)
        Y = reward + (current_model.gamma * maxQ)
        Y = torch.Tensor([Y]).detach()
        loss = loss_fn(X, Y)
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.losses.append(loss.item())
        current_model.optimizer.step()
        current_model.iterations += 1
        if current_model.iterations % current_model.sync_freq == 0:
            targ_network.load_state_dict(current_model.state_dict())

    def initialize_target_network(self, model):
        target_network = deepcopy(model)
        target_network.load_state_dict(model.state_dict())
        return target_network

 
    def get_target_network(self, model):
        return self.target_dictionary[model.turn] 
