from abc import abstractmethod
from .. import constants as c
from env import GameState


class Agent:
    def __init__(self, agentType=c.MOUSE_AGENT):
        self.agentType = agentType

    @abstractmethod
    def getAction(self, state: GameState):
        """abstract method"""

    @abstractmethod
    def reflex(self, state: GameState):
        """abstract method"""


class RandomAgent(Agent):
    def getAction(self, state: GameState):
        ...


class LocalAgent(Agent):
    def getAction(self, state: GameState):
        # TODO
        ...

    def reflex(self, state: GameState):
        # TODO
        ...

    # some functions


class DQNAgent(Agent):
    def getAction(self, state: GameState):
        # TODO
        ...

    def reflex(self, state: GameState):
        # TODO
        ...

    # some functions

