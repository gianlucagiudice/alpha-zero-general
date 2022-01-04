from src.connect4.Connect4Game import Connect4Game
from src.connect4.Connect4Players import RandomConnect4Player, HumanConnect4Player
from src.Arena import Arena
from src.connect4.keras.NNet import NNetWrapper
from src.MCTS import MCTS
from src.connect4.Connect4Players import MCTSConnect4Player


def main():
    connect4game = Connect4Game()
    player1 = HumanConnect4Player(connect4game, 1)

    # nnet players
    nnet = NNetWrapper(connect4game)
    nnet.load_checkpoint('./temp', 'best.h5')
    args = {'numMCTSSims': 20, 'cpuct': 1.0}
    mcts = MCTS(connect4game, nnet, args)

    player = MCTSConnect4Player(mcts)


    print("\t\t===== CONNECT 4 GAME=====")
    arena = Arena(player1.play, player.play, connect4game)

    arena.playGame(show=True, verbose=True)


main()
