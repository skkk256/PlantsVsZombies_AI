from ..tool import Control
from .. import constants as c
from ..state.level import Level
import pygame as pg


class GameState(Level):
    def __init__(self):
        super().__init__()

    def __str__(self):
        gameState = self.getGameState()
        sun_value = gameState["sun_value"]
        plant_avail = gameState["plant_availability"]
        grid_state =gameState['grid_state']
        grid_state_fstr = ""
        for i in range(len(grid_state)):
            grid_state_fstr += f"Row {i}:"
            for j in range(len(grid_state[i])):
                grid_state_fstr += f"({grid_state[i][j][0][0]:<10}, {grid_state[i][j][0][1]:<2}),  {grid_state[i][j][1]:<2}"

            grid_state_fstr += "\n"

        return f"""
        game_info: {self.game_info},
        map_data: {self.map_data},
        sun_value: {sun_value},
        plant_avail: {plant_avail},
        grid_state:\n{grid_state_fstr}
        """

    def getGameState(self):
        grid_state = [[[("Blank", 0), 0] for _ in range(10)] for _ in range(5)]
        for row in self.plant_groups:
            plants_dict = row.spritedict
            for plant, rect in plants_dict.items():
                grid_index = self.map.getMapIndex(int(rect.x + rect.width / 2), int(rect.y + rect.height / 2))
                grid_state[grid_index[1]][grid_index[0]][0] = (plant.name, plant.health)

        for row in self.zombie_groups:
            zombies = row.spritedict
            for zombie, rect in zombies.items():
                grid_index = self.map.getMapIndex(int(rect.x), int(rect.y + rect.height))
                grid_state[grid_index[1]][grid_index[0]][1] = zombie.health

        return {
            "sun_value": self.menubar.sun_value,
            "grid_state": grid_state,
            "plant_availability": self.menubar.getAvailability()
        }

    def updateByAction(self, current_time, action):
        ...


class GameRunner(Control):
    def __init__(self):
        super().__init__()
        self.agent = None

    def setup_states(self, state_dict, start_state, agent):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
        self.agent = agent
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

    def updateByAction(self):
        self.current_time = pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        self.state.updateByAction(self.current_time, self.agent.getAction(self.state))
        self.agent.reflex(self.state)

    def update(self, is_render=True):
        if self.agent.agentType == c.MOUSE_AGENT:
            self.updateByMouse()
            self.render()
        else:
            self.updateByAction()
            if is_render:
                self.render()

    def main(self):
        while not self.done:
            self.event_loop()
            self.update(True)
            pg.display.update()
            self.clock.tick(self.fps)
        print('game over')

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                print(self.state)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            elif event.type == pg.MOUSEBUTTONDOWN and self.agent.agentType == c.MOUSE_AGENT:
                self.mouse_pos = pg.mouse.get_pos()
                self.mouse_click[0], _, self.mouse_click[1] = pg.mouse.get_pressed()