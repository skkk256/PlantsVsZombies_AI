from abc import abstractmethod
from .. import constants as c
from source.agents.env import GameState
import random


class Action:
    def __init__(self, plant_name, cost, x, y):
        self.plant_cost = cost
        self.plant_name = plant_name
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.plant_name}, ({self.x}, {self.y})"


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
            return
        sun_value = gameState["sun_value"]
        plant_availability = gameState["plant_availability"]  # [(plant_name, frozen_time, sun_cost), ..., ]
        grid_state = gameState["grid_state"] # 5*10 list, entry: [ (plant_name, hp), zombie_hp ]

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
                return Action(plant[0], plant[2], coordinate[0], coordinate[1])
            else:
                return Action(c.IDLE, 0, 0, 0)
        else:
            return Action(c.IDLE, 0, 0, 0)



class LocalAgent(Agent):
    def getAction(self, state: GameState, current_time):
        # TODO
        ...

    def reflex(self, state: GameState):
        # TODO
        ...

    # some functions


class DQNAgent(Agent):
    def getAction(self, state: GameState, current_time):
        # TODO
        ...

    def reflex(self, state: GameState):
        # TODO
        ...

    # some functions

