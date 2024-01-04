from ..tool import Control
from .. import constants as c
from ..state.level import Level
from abc import abstractmethod
import pygame as pg


class GameState(Level):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"""
        game_info: {self.game_info},
        map_data: {self.map_data},
        sun_value: {self.menubar.sun_value},
        plant_groups: {self.plant_groups[0]},
        zombie_groups: {self.zombie_groups[0]}
        """


class Agent:
    def __init__(self, agentType=c.MOUSE_AGENT):
        self.agentType = agentType

    @abstractmethod
    def getAction(self, state: GameState):
        """abstract method"""

    @abstractmethod
    def reflex(self, state: GameState):
        """abstract method"""


class GameRunner(Control):
    def __init__(self):
        super().__init__()
        self.agent = None

    def setup_states(self, state_dict, start_state, agent):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
        self.agent: Agent = agent
        self.state.startup(self.current_time, self.game_info)


    def render(self):
        self.state.render(self.screen)

    def updateByMouse(self):
        self.current_time = pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        self.state.update(self.current_time, self.mouse_pos, self.mouse_click)
        self.mouse_pos = None
        self.mouse_click[0] = False
        self.mouse_click[1] = False

    def updateByAPI(self):
        self.current_time = pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        self.state.updateByAPI(self.current_time, *self.agent.getAction(self.state))
        self.agent.reflex(self.state)

    def update(self, is_render=True):
        if self.agent.agentType == c.MOUSE_AGENT:
            self.updateByMouse()
            self.render()
        else:
            self.updateByAPI()
            if is_render:
                self.render()

    def main(self):
        while not self.done:
            self.event_loop()
            self.update(True)
            pg.display.update()
            self.clock.tick(self.fps)
        print('game over')