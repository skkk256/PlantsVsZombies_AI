__author__ = 'marble_xu'

from . import constants as c
from .state import mainmenu, screen, level
from .agents.env import GameState, GameRunner
from .agents.agent import Agent


def main():
    # game = tool.Control()
    game = GameRunner()
    state_dict = {c.MAIN_MENU: mainmenu.Menu(),
                  c.GAME_VICTORY: screen.GameVictoryScreen(),
                  c.GAME_LOSE: screen.GameLoseScreen(),
                  c.LEVEL: GameState()}
    agent = Agent(c.MOUSE_AGENT)
    game.setup_states(state_dict, c.LEVEL, agent)
    game.main()
