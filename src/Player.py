from abc import ABC, abstractmethod
import numpy as np

from src.Game import Game


class Player(ABC):
    def __init__(self, game: Game, player: int):
        self.game = game
        self.player = player

    @abstractmethod
    def play(self, board: np.array) -> int:
        pass
