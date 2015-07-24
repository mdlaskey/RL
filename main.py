from race_game import RaceGame
import pygame

race_game = RaceGame(graphics=True, MAX_LAPS=3)
while race_game.running:
    race_game.run_frame()
    race_game.control_car(driving_agent=True)



