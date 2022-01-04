import train_connect4
from src.connect4.Connect4Game import Connect4Game
from src.connect4.Connect4Players import HumanConnect4Player
from src.Arena import Arena
from src.connect4.keras.NNet import NNetWrapper
from src.MCTS import MCTS
from src.connect4.Connect4Players import MCTSConnect4Player
import os


def main():
    # Init Game
    connect4game = Connect4Game()
    # Init players
    human_player = HumanConnect4Player(connect4game, 1)
    rl_player = init_rl_player(connect4game)
    # Init arena
    arena = Arena(human_player.play, rl_player.play, connect4game)
    # Start the game
    print("\t\t===== CONNECT 4 GAME=====")
    arena.playGame(show=True, verbose=True)


def init_rl_player(game):
    # Init neural network
    nnet = NNetWrapper(game)
    # Load best weights
    nnet.load_checkpoint('temp', 'best.h5')
    # Init Monte Carlo tree search
    mcts = MCTS(game, nnet, train_connect4.args)
    # Return player
    return MCTSConnect4Player(game, -1, mcts)


if __name__ == '__main__':
    main()
