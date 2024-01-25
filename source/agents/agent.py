from abc import abstractmethod
from .. import constants as c
from source.agents.env import GameState, process_state
import random
import queue
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from .env import Action

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
    
def decode_action(action_id):
    total_columns = 9
    total_rows = 5
    total_plants = 4

    # Calculate the plant_id, column, and row
    plant_id = action_id % total_plants
    column = (action_id // total_plants) % total_columns
    row = action_id // (total_plants * total_columns)
    
    return row, column, plant_id

def encode_action(action):
    plant_name = action.plant_name
    if plant_name == c.SUNFLOWER:
        plant_id = 0
    elif plant_name == c.CHERRYBOMB:
        plant_id = 1
    elif plant_name == c.PEASHOOTER:
        plant_id = 2
    elif plant_name == c.WALLNUT:
        plant_id = 3
    else:
        return 180
    x = action.x
    y = action.y
    
    action_id = 36*x + 4*y + plant_id
    
    return action_id



class Agent:
    def __init__(self, agentType=c.MOUSE_AGENT):
        self.agentType = agentType
        self.play_time = 0
        self.play_interval = 1000

    @abstractmethod
    def getAction(self, state: GameState, current_time):
        """abstract method"""

    @abstractmethod
    def reflex(self, state: GameState):
        """abstract method"""


class RandomAgent(Agent):
    def getAction(self, state: GameState, current_time):
        gameState = state.getGameState()
        if gameState == "end":
            return Action(c.IDLE, 0, 0, 0)
        sun_value = gameState["sun_value"]
        plant_availability = gameState["plant_availability"]  # [(plant_name, frozen_time, sun_cost), ..., ]
        grid_state = gameState["grid_state"] # 5*10 list, entry: [ (plant_name, hp), zombie_hp ]
        #print(grid_state)
        if current_time - self.play_time >= self.play_interval:
            self.play_time = current_time
            available_plant = []
            available_coordinate = []
            for plant_data in plant_availability:
                if plant_data[1] == 0 and sun_value > plant_data[2]:
                    available_plant.append(plant_data)
            for i in range(5):
                for j in range(9):
                    if grid_state[i][j][0][0] == c.BLANK:
                        available_coordinate.append((i, j))
            if len(available_coordinate) > 0 and len(available_plant) > 0 and random.random() < 0.5:
                plant = random.choice(available_plant)
                coordinate = random.choice(available_coordinate)
                #print(Action(plant[0], plant[2], coordinate[0], coordinate[1]))
                return Action(plant[0], plant[2], coordinate[0], coordinate[1])
            else:
                #print(c.IDLE, 0, 0, 0)
                return Action(c.IDLE, 0, 0, 0)
        else:
            #print(c.IDLE, 0, 0, 0)
            return Action(c.IDLE, 0, 0, 0)


class LocalAgent(Agent):
    def getAction(self, state: GameState, current_time):
        gameState = state.getGameState()
        if gameState == "end":
            return
        sun_value = gameState["sun_value"]
        plant_availability = gameState["plant_availability"]  # [(plant_name, frozen_time, sun_cost), ..., ]
        grid_state = gameState["grid_state"] # 5*10 list, entry: [ (plant_name, hp), zombie_hp ]
        discount = 0.8
        if current_time - self.play_time < self.play_interval:
            return Action(c.IDLE, 0, 0, 0)
        self.play_time = current_time
        available_plant = []
        available_coordinate = []
        total_sunflowers = 0
        for plant_data in plant_availability:
            if plant_data[1] == 0 and sun_value > plant_data[2]:
                available_plant.append(plant_data)
        plants_count = [{plant[0]:0 for plant in plant_availability} for __ in range(5)]
        #First we compute each line the index 0 box to check whether the value is less than some fix threshold
        choose = 0
        least = 99999999999
        gama = 0.9
        plants = {each[0]:(each[1], each[2]) for each in plant_availability}
        values = [[0 for __ in range(9)] for _ in range(5)]
        for i in range(5):
            value = 0
            #For this process, we tranverse from right to left for each single line
            for j in range(8, -1, -1):
                grid = grid_state[i][j]
                plant_name = grid_state[i][j][0][0]
                #there is a discount factor gama, the equation is: V'(t) = V(t) + gama*V(t+1), we then use value iteration to compute value
                value *= gama
                
                if plant_name != c.BLANK:
                    # pdb.set_trace()
                    plants_count[i][plant_name] += 1
                if plant_name == c.SUNFLOWER:
                    total_sunflowers += 1
                if plant_name == c.BLANK:
                    value += grid[1]
                if plant_name == c.WALLNUT:
                    value /= grid[0][1]/10
                if plant_name == c.PEASHOOTER:
                    value -= 10
                if value < 0:
                    value = 0
                values[i][j] = value
        values = np.array(values)
        
        
        max_value = max(values[:, 0])
        min_value = min(values[:, 0])
            
        #If bigger, then turn into defence action
        thres = 5
        if max_value >= thres or total_sunflowers >= 10:
            store = queue.PriorityQueue()
            choose_line = 0
            least = 0
            for i in range(5):
                #First we choose a line that's most dangerous to handle
                if values[i][0] > least:
                    choose_line = i
                    least = values[i][0]
            #If there are lines that have zombie we can't kill, then plant peashoter first
            if least > 0:
                for j in range(9):
                    if grid_state[choose_line][j][0][0] == c.BLANK:
                        #After planting the peashoter or wallnut, we first try to compare the index 0 value
                        new_line_value = [0 for ___ in range(9)]
                        #Peashoter
                        for k in range(9):
                            grid = grid_state[choose_line][k]
                            plant_name = grid_state[choose_line][k][0][0]
                            value = 0
                            #there is a discount factor gama, the equation is: V'(t) = V(t) + gama*V(t+1), we then use value iteration to compute value
                            value *= gama
                            if plant_name != c.BLANK:
                                # pdb.set_trace()
                                plants_count[choose_line][plant_name] += 1
                            if plant_name == c.SUNFLOWER:
                                total_sunflowers += 1
                            if plant_name == c.BLANK:
                                value += grid[1]
                            if plant_name == c.WALLNUT:
                                value /= grid[0][1]/30
                            if plant_name == c.PEASHOOTER:
                                value -= 10
                            if k == j:
                                value -= 10
                            if value < 0:
                                value = 0
                            new_line_value[k] = value
                        if values[choose_line][0] - new_line_value[0] > 0:
                            store.put((values[choose_line][0] - new_line_value[0], c.PEASHOOTER, j, choose_line))
                        #If all values are equal to zero, then we use the average value among one single line
                        #Also if this has no change, then we try to plant the peashoter
                #If we dicide to plant peashoter
                if store.empty() == False:
                    action = store.get()
                    if action[1] == c.PEASHOOTER and sun_value >= 100 and plants[c.PEASHOOTER][0] == 0:
                        return Action(c.PEASHOOTER, 100, action[2], action[3])
            
            #Else we dicide to plant wallnut
            #Similar to before, we assume each blank box to have a wallnut, then compare the average value in a line to dicide where to put
            else:
                zombies = [0 for _ in range(5)]
                for i in range(5):
                    for j in range(9):
                        zombies[i] += grid_state[i][j][1]
                lease = 99999
                choose_line = 0
                for i in range(5):
                    if lease < zombies[i]:
                        lease = zombies[i]
                        choose_line = i
                for j in range(9):
                    if grid_state[choose_line][j][1] > 0 and sun_value >= 50 and plants[c.WALLNUT][0] == 0:
                        return Action(c.WALLNUT, 50, j, choose_line)
                
            return Action(c.IDLE, 0, 0, 0)
        #Else smaller we turn into preparation action
        else:
            #If we enter preparation action, we choose a line that has the smallest sunflowers or the least dangerous, to plant a sunflower
            store = queue.PriorityQueue()
            for i in range(5):
                for j in range(9):
                    if grid_state[i][j][0][0] == c.BLANK:
                        store.put((j+values[i][j], (i, j)))
                        break
            
            if sun_value >=50 and plants[c.SUNFLOWER][0] == 0:
                place = store.get()[1]
                return Action(c.SUNFLOWER, 50, place[1], place[0])
            else:
                return Action(c.IDLE, 0, 0, 0)
    
    def reflex(self, state: GameState):
        # TODO
        ...

    # some functions


class DQNAgent(Agent):
    def __init__(self, agentType=c.DQN_AGENT, epsilon=0.01):
        super(DQNAgent, self).__init__(agentType)
        self.model = DQN(129, 181)
        self.model.load_state_dict(torch.load('source/agents/dqn_model.pth'))
        self.epsilon = epsilon
    def act(self, state: GameState):
        state = state.getGameState()
        if(type(state) != dict):
            return Action(c.IDLE, 0, 0, 0)
        plant_availability = state["plant_availability"]
        grid_state = state["grid_state"]
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
        

        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(process_state(state)).unsqueeze(0)
                q_value = self.model.forward(state)
                q_value_legal = q_value.squeeze()[legal_actions]
                action = legal_actions[q_value_legal.argmax(0)]
        else:
            action = random.choice(legal_actions)
        
        x, y, plant = decode_action(action)
        return Action(plant_availability[plant][0], plant_availability[plant][2], x, y)

    def getAction(self, state: GameState, current_time):
        gameState = state.getGameState()
        if gameState == "end":
            return Action(c.IDLE, 0, 0, 0)
        if current_time - self.play_time >= self.play_interval:
            self.play_time = current_time
            act = self.act(state)
            return act
        else:
            return Action(c.IDLE, 0, 0, 0)
