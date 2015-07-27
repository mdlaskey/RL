from race_game import RaceGame
from Agents.DAgger import Dagger
import pygame
import car
import dummy_car

if __name__ == '__main__':
    race_game = RaceGame(agent = Dagger(),graphics=True, MAX_LAPS=3)
    while race_game.running:
        race_game.control_car(input_sequence=None, driving_agent=True)



