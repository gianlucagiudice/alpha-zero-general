from typing import Callable

import logging

from tqdm import tqdm

from colorama import Fore, Style

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1: Callable, player2: Callable, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False, show=False):
        # Init board
        curr_player = 1
        board = self.game.getInitBoard()
        turn = 0

        # Show game
        self.display_game(board, turn, curr_player, verbose, show)

        # Gaming loop
        while self.game.getGameEnded(board, curr_player) == 0:
            turn = turn + 1
            # Play turn
            for player in [self.player1, self.player2]:
                # Player make a choice
                choice = player(board)
                board, curr_player = self.game.getNextState(board, curr_player, choice)
                # Show game
                self.display_game(board, turn, -curr_player, verbose, show)
                # Won?
                winner_player = self.game.getGameEnded(board, curr_player)
                if winner_player != 0:
                    break

        # Return winner
        the_winner_is = curr_player * self.game.getGameEnded(board, curr_player)
        if the_winner_is not in [1, -1]:
            the_winner_is = 0

        if verbose:
            print(f"\n>>> Game over! Turn {turn}, the winner is {the_winner_is} <<< ")
            input("(Press any key to continue...)")

        return the_winner_is


    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

    def display_game(self, board, turn, player, verbose, show):
        if show:
            self.game.show(board)
        if verbose:
            if turn == 0:
                print(f'\n>>> Starting position <<<\n')
            else:
                color = Fore.RED if player == 1 else Fore.BLUE
                print(f"\n>>> Turn {turn}; {color}Player {player}{Style.RESET_ALL} <<<\n")
            self.game.display(board)
