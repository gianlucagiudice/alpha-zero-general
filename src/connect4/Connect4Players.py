import numpy as np
from src.connect4.Connect4Game import Connect4Game
from src.Player import Player
from src.MCTS import MCTS

class RandomConnect4Player(Player):
    def __init__(self, game: Connect4Game, player: int):
        self.game = game
        self.player = player

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player)
        valid_actions = np.arange(valid_moves.size)[valid_moves]
        return np.random.choice(valid_actions)


class HumanConnect4Player(Player):
    def __init__(self, game, player):
        self.game = game
        self.player = player

    def play(self, board: np.array) -> int:
        valid_moves = self.game.getValidMoves(board, self.player)
        print('List of valid moves: ', [i for i, valid in enumerate(valid_moves) if valid])
        while True:
            try:
                move = int(input('Choose your move: '))
                if valid_moves[move]:
                    break
                else:
                    print('Invalid move.')
            except ValueError:
                print('Invalid move.')
        return move


class MCTSConnect4Player:
    def __init__(self, game, player, mcts: MCTS, verbose=False):
        self.game = game
        self.player = player
        self.mcts = mcts
        self.verbose = verbose

    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=0))
