import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from .env import GameState, GameRunner, process_state, Action
from .. import constants as c
from ..state import mainmenu, screen
from .agent import DQNAgent, RandomAgent, decode_action, encode_action
import math
import pygame as pg
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
    def act(self, state, epsilon=0.1):
        plant_availability = state["plant_availability"]  # [(plant_name, frozen_time, sun_cost), ..., ]
        grid_state = state["grid_state"] # 5*10 list, entry: [ (plant_name, hp), zombie_hp ]
        sun_value = state["sun_value"]
        available_plants = [plant for plant in plant_availability if plant[1] == 0 and sun_value >= plant[2]]
        available_coordinates = [(i, j) for i in range(5) for j in range(9) if grid_state[i][j][0][0] == c.BLANK]
        
        legal_actions = []
                
        for plant in available_plants:
            plant_name = plant[0]
            plant_id = None
            if plant_name == c.SUNFLOWER:
                plant_id = 0
            elif plant_name == c.CHERRYBOMB:
                plant_id = 1
            elif plant_name == c.PEASHOOTER:
                plant_id = 2
            elif plant_name == c.WALLNUT:
                plant_id = 3
        
            if plant_id is not None:
                for coord in available_coordinates:
                    action_id = 36*coord[0] + 4*coord[1] + plant_id
                    legal_actions.append(action_id)
        
        if len(legal_actions) == 0:
            return Action(c.IDLE, 0, 0, 0)

        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(process_state(state)).unsqueeze(0)
                q_value = self.forward(state)
                q_value_legal = q_value.squeeze()[legal_actions]
                action = legal_actions[q_value_legal.argmax(0)]
        else:
            action = random.choice(legal_actions)
        
        x, y, plant = decode_action(action)
        return Action(plant_availability[plant][0], plant_availability[plant][2], x, y)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

'''for action_id in range(180):
    row, column, plant_id = decode_action(action_id)
    print(f"Action ID {action_id} => Row: {row}, Column: {column}, Plant ID: {plant_id}")
'''

# Hyperparameters
batch_size = 32
gamma = 0.99
replay_buffer = ReplayBuffer(50000)
model = DQN(129, 181)
target_model = DQN(129, 181)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.MSELoss()
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 1000
epsilon_by_episode = lambda episode_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode_idx / epsilon_decay)
TARGET_UPDATE_FREQUENCY = 1000

torch.save(model.state_dict(), 'source/agents/dqn_model.pth')

def update_target_model():
    target_model.load_state_dict(model.state_dict())

def update_model():
    if len(replay_buffer) < batch_size:
        return
    
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)
    
    q_values = model(state)
    next_q_values = target_model(next_state)
    
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = loss_fn(q_value, expected_q_value)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.save(model.state_dict(), 'source/agents/dqn_model.pth')

game = GameRunner(20, 4)
state_dict = {c.MAIN_MENU: mainmenu.Menu(),
            c.GAME_VICTORY: screen.GameVictoryScreen(),
            c.GAME_LOSE: screen.GameLoseScreen(),
            c.LEVEL: GameState()}
average_rewards = []

window_size = 50

for episode in range(10000):
    act_list = []
    epsilon = epsilon_by_episode(episode)

    game.setup_states(state_dict, c.LEVEL, DQNAgent(c.DQN_AGENT, epsilon))
    state = game.state.getGameState()
    print("episode:", episode, "epsilon:", epsilon)
    rolling_sum = 0
    tot_reward = 0
    max_duaration = 1000000
    while not game.done and game.time < max_duaration:
        game.event_loop()
        action, next_state, reward, done = game.update()
        pg.display.update()
        game.clock.tick(game.fps)
        game.time+=1
        if not game.done:
            replay_buffer.push(process_state(state), encode_action(action), reward, process_state(next_state), done)
        tot_reward += reward
    
    update_model()
    rolling_sum += tot_reward
    if (episode + 1) % window_size == 0:
        average_reward = rolling_sum / window_size
        average_rewards.append(average_reward)
        rolling_sum = 0
    
    print("tot_time:", game.time)
    print("tot_reward:", tot_reward)
    game.done = True
    game.time = 0
    print('game over')
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        update_target_model()


torch.save(model.state_dict(), 'source/agents/dqn_model.pth')
plt.plot(range(window_size, 10001, window_size), average_rewards)
plt.title(f'Average Total Reward per {window_size} Episodes')
plt.xlabel('Episode Index')
plt.ylabel('Average Total Reward')
plt.show()