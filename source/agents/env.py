from ..tool import Control
from .. import constants as c
from ..state.level import Level
import pygame as pg
from ..component import map, plant, zombie, menubar
import random


class Action:
    def __init__(self, plant_name, cost, x, y):
        self.plant_cost = cost
        self.plant_name = plant_name
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.plant_name}, ({self.x}, {self.y})"


def process_state(state):
    if type(state) != dict:
        return
    sun_value = state["sun_value"]
    grid_state = state["grid_state"]
    plant_availability = state["plant_availability"]

    grid_flat = []
    
    row_sunflower_count = [0] * len(grid_state)  # Number of Sunflowers per row
    row_peashooter_count = [0] * len(grid_state)  # Number of Peashooters per row
    row_wallnut_count = [0] * len(grid_state)  # Number of Wall-nuts per row
    row_zombie_total_hp = [0] * len(grid_state)  # Total Zombie HP per row

    for row_index, row in enumerate(grid_state):
        for cell in row:
            plant_info = cell[0]
            zombie_hp = cell[1]
            plant_hp = plant_info[1] if plant_info[0] != c.BLANK else 0
            grid_flat.append(plant_hp)
            grid_flat.append(zombie_hp)
            if plant_info[0] == c.SUNFLOWER:
                row_sunflower_count[row_index] += 1
            elif plant_info[0] == c.PEASHOOTER:
                row_peashooter_count[row_index] += 1
            elif plant_info[0] == c.WALLNUT:
                row_wallnut_count[row_index] += 1
            row_zombie_total_hp[row_index] += zombie_hp
            
    plants_info = []
    for plant in plant_availability:
        avaliable = 1 if plant[1] == 0 and sun_value >= plant[2] else 0
        frozen_time = plant[1]
        plants_info.append(avaliable)
        plants_info.append(frozen_time)

    state_processed = (
        grid_flat +
        plants_info +
        row_sunflower_count +
        row_peashooter_count +
        row_wallnut_count +
        row_zombie_total_hp +
        [sun_value]
    )
    
    return state_processed

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
        grid_state = [[[(c.BLANK, 0), 0] for _ in range(10)] for _ in range(5)]
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

    def setupCars(self):
        self.cars = []

    def updateByAction(self, surface, current_time, action):
        self.current_time = self.game_info[c.CURRENT_TIME] = current_time
        map_x, map_y = action.y, action.x
        plant_name = action.plant_name
        plant_cost = action.plant_cost
        x, y = self.map.getMapGridPos(map_x, map_y)
        if self.map.isValid(map_x, map_y) and self.map.isMovable(map_x, map_y) and plant_name != c.IDLE:
            self.addPlant(plant_name, plant_cost, x, y, map_x, map_y)

        self.updateEnvironment()

        self.draw(surface)
        return action

    def updateEnvironment(self):
        if self.zombie_start_time == 0:
            self.zombie_start_time = self.current_time

        if len(self.zombie_list) > 0:
            data = self.zombie_list[0]
            if data[0] <= (self.current_time - self.zombie_start_time):
                self.createZombie(data[1], data[2])
                self.zombie_list.remove(data)

        for i in range(self.map_y_len):
            self.bullet_groups[i].update(self.game_info)
            self.plant_groups[i].update(self.game_info)
            self.zombie_groups[i].update(self.game_info)
            self.hypno_zombie_groups[i].update(self.game_info)
            for zombie in self.hypno_zombie_groups[i]:
                if zombie.rect.x > c.SCREEN_WIDTH:
                    zombie.kill()

        self.head_group.update(self.game_info)
        self.sun_group.update(self.game_info)

        if self.produce_sun:
            if (self.current_time - self.sun_timer) > c.PRODUCE_SUN_INTERVAL:
                self.sun_timer = self.current_time
                map_x, map_y = self.map.getRandomMapIndex()
                x, y = self.map.getMapGridPos(map_x, map_y)
                self.sun_group.add(plant.Sun(x, 0, x, y))
        
        for sun in self.sun_group:
            if sun.getSun():
                self.menubar.increaseSunValue(sun.sun_value)

        for car in self.cars:
            car.update(self.game_info)

        self.menubar.update(self.current_time)

        self.checkBulletCollisions()
        self.checkZombieCollisions()
        self.checkPlants()
        self.checkCarCollisions()
        self.checkGameState()


class GameRunner(Control):
    def __init__(self, game_speed, game_level):
        super().__init__()
        self.agent = None
        self.speed_up_multiplier = game_speed
        self.fps = self.speed_up_multiplier * 60
        self.game_info = {c.CURRENT_TIME: 0.0,
                          c.LEVEL_NUM: game_level}

    def setup_states(self, state_dict, start_state, agent):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
        self.agent = agent
        self.state.startup(self.current_time, self.game_info)
        self.time = 0
        self.done = False

    def updateByMouse(self):
        self.state.update(self.screen, self.current_time, self.mouse_pos, self.mouse_click)
        self.mouse_pos = None
        self.mouse_click[0] = False
        self.mouse_click[1] = False

    def updateByAction(self):
        action = self.agent.getAction(self.state, self.current_time)
        self.state.updateByAction(self.screen, self.current_time, action)
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
        
        next_state = self.state.getGameState()
        reward = 0
        self.die_zombies = 0
        if(type(next_state) != dict):
            self.done = True
        else:
            base_reward = 10
            reward += base_reward
            
            self.tot_zombie = sum(1 for i in range(self.state.map_y_len) for zombie in self.state.zombie_groups[i] if zombie.state != c.DIE)
            die_zombies = sum(1 for i in range(self.state.map_y_len) for zombie in self.state.zombie_groups[i] if zombie.state == c.DIE)
            reward += die_zombies * 75 
            
            if action.plant_name == c.WALLNUT and action.y < 3:
                reward -= 10
            if action.plant_name == c.PEASHOOTER and action.y > 5:
                reward -= 20
            if action.plant_name == c.CHERRYBOMB:
                affected_zombies = sum(1 for i in range(self.state.map_y_len) for j in range(max(0, i - 1), min(self.state.map_y_len, i + 2)) for zombie in self.state.zombie_groups[j] if zombie.state == c.WALK)
                reward += affected_zombies * 50
                reward -= 100
            
            for i in range(self.state.map_y_len):
                zombie_count = len([zombie for zombie in self.state.zombie_groups[i] if zombie.state == c.WALK])
                sunflower_count = sum(1 for plant in self.state.plant_groups[i] if plant.name == c.SUNFLOWER)
                peashooter_count = sum(1 for plant in self.state.plant_groups[i] if plant.name == c.PEASHOOTER)
                
                if zombie_count > 0 and action.plant_name == c.PEASHOOTER and action.x == i:
                    reward += 20

                if zombie_count == 0 and action.plant_name == c.SUNFLOWER and action.x == i:
                    reward += 10

                if zombie_count == 0 and action.plant_name == c.WALLNUT and action.x == i:
                    reward -= 10
                
                reward += (sunflower_count * 5) + (peashooter_count * 10)

        return action, next_state, reward, self.done
        
    def update(self):
        self.current_time = self.speed_up_multiplier * pg.time.get_ticks()
        if self.state.done:
            self.flip_state()
        if self.agent.agentType == c.MOUSE_AGENT:
            self.updateByMouse()
            return Action(c.IDLE, 0, 0, 0), None, 0, False
        else:
            return self.updateByAction()

    def main(self):
        max_duaration = 100000
        while not self.done and self.time < max_duaration:
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)
            self.time+=1
        #self.done = True
        #self.time = 0
        print('game over')

    def event_loop(self):
        # print(self.current_time)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                #print(self.state)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            elif event.type == pg.MOUSEBUTTONDOWN and self.agent.agentType == c.MOUSE_AGENT:
                self.mouse_pos = pg.mouse.get_pos()
                self.mouse_click[0], _, self.mouse_click[1] = pg.mouse.get_pressed()