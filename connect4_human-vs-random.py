from src.connect4.Connect4Game import Connect4Game
from src.connect4.Connect4Players import RandomConnect4Player, HumanConnect4Player
from src.Arena import Arena


def main():
    connect4game = Connect4Game()
    player1 = HumanConnect4Player(connect4game, 1)
    player2 = RandomConnect4Player(connect4game, -1)

    print("\t\t===== CONNECT 4 GAME=====")
    arena = Arena(player1.play, player2.play, connect4game)

    arena.playGame(show=True, verbose=True)


main()
