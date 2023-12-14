import pygame
import numpy as np
import torch
from agent import Agent
from copy import deepcopy
from matplotlib import pylab as plt
import torch.optim as optim

pygame.init()

ROW_COUNT = 6
COLUMN_COUNT = 7

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

RED_PIECE = 1
YELLOW_PIECE = 2



def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
        
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True

    return False

def drop_piece(board, row, col, piece):
    board[row][col] = piece


def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == RED_PIECE:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == YELLOW_PIECE: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()
turns = {
    0 : 1,
    1 : 2
}

num_not_finished = 0

Agent = Agent()
Agent.first_turn.scheduler = optim.lr_scheduler.StepLR(Agent.first_turn.optimizer, step_size=500, gamma=0.1)
Agent.second_turn.scheduler = optim.lr_scheduler.StepLR(Agent.second_turn.optimizer, step_size=500, gamma=0.1)
epochs = 3_000
for i in range(epochs):
    board = create_board()
    game_over = False
    turn = 0
    screen = pygame.display.set_mode(size)
    draw_board(board)
    pygame.display.update()
    print(i)
    # if i > 8000 and len(Agent.first_turn.losses) != 0 and Agent.first_turn.losses[-1] < 120:
    #     Agent.first_turn.save(file_name='first_turn.pth')
    #     break
    while not game_over:
        acting_model, waiting_model = (Agent.first_turn, Agent.second_turn) if turn == 0 else (Agent.second_turn, Agent.first_turn)
        if len(Agent.states) == 1:
            action, acting_model_predicted_qvals = Agent.states.pop()
            X = acting_model_predicted_qvals.squeeze()[action]
        else:
            state_ = board.reshape(1,42) + np.random.rand(1,42)/10.0
            state = torch.from_numpy(state_).float()
            action, acting_model_predicted_qvals = Agent.action(state, acting_model)
            X = acting_model_predicted_qvals.squeeze()[action]
        if is_valid_location(board, action):
            row = get_next_open_row(board, action)
            drop_piece(board, row, action, turns[turn])
            if winning_move(board, turns[turn]):
                game_over = True
                Agent.reward_winning_move(acting_model, acting_model_predicted_qvals.squeeze()[action], turn)
                waiting_model_predicted_qval = torch.max(waiting_model(state).squeeze())
                Agent.punish_losing_move(waiting_model, waiting_model_predicted_qval, turn)
            else:
                state_ = board.reshape(1,42) + np.random.rand(1,42)/10.0
                state = torch.from_numpy(state_).float()
                action, waiting_model_predicted_qval = Agent.action(state, waiting_model)
                Agent.states.append((action,waiting_model_predicted_qval))
                board_copy = deepcopy(board)
                if is_valid_location(board_copy, action):
                    row = get_next_open_row(board_copy, action)
                    drop_piece(board_copy, row, action, turns[not turn])
                    reward = -10 if winning_move(board_copy, turns[not turn]) else -1
                    board_copy = board_copy.reshape(1,42) + np.random.rand(1,42)/10.0
                    board_copy = torch.from_numpy(board_copy).float()
                    Agent.calculate_q(board_copy, reward, X, acting_model, turn)
            draw_board(board)
            turn ^= 1
        else: # need to punish neural networks for trying to make moves that DONT FUCKING WORK
            Agent.punish_losing_move(acting_model, acting_model_predicted_qvals, turn)
            num_not_finished += 1
            game_over = True
    if Agent.first_turn.epsilon > 0.1: #R
        Agent.first_turn.epsilon -= (1/epochs)
    if Agent.second_turn.epsilon > 0.1:
        Agent.second_turn.epsilon -= (1/epochs)
    
print(num_not_finished)
plt.figure(figsize=(10,7))
plt.plot(Agent.first_turn.losses)
plt.xlabel("Epochs",fontsize=22)
plt.ylabel("Loss",fontsize=22)
plt.savefig("loses.png")
plt.show()
