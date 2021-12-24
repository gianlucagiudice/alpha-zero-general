import numpy as np
from src.connect4.Connect4Game import Connect4Game
from src.Player import Player


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


class OneStepLookaheadConnect4Player:
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose: print('Playing random action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))

        return ret_move
