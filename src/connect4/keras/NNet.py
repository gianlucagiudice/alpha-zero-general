import visualkeras

from src.NeuralNet import NeuralNet

from src.connect4.keras.Connect4NNet import Connect4NNet
from multiprocessing import cpu_count

import numpy as np
import os

args = dict(
    lr=0.001,
    dropout=0.3,
    epochs=10,
    batch_size=64,
    cuda=False,
    num_channels=512,
)


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        super().__init__(game)
        self.nnet = Connect4NNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        # History of training
        self.last_training_history = None

    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        # Input and output for training
        x, y = input_boards, [target_pis, target_vs]
        # Traing
        self.last_training_history = self.nnet.model.fit(
            x=x, y=y, batch_size=args["batch_size"], epochs=args["epochs"],
            use_multiprocessing=True, workers=cpu_count()
        )

    def predict(self, board):
        # Preprocess input
        board = board[np.newaxis, :, :]
        # Prediction on board
        pi, v = self.nnet.model.predict(board)
        # Return prediction
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)

    def save_plot_network(self, folder='report/images', filename='my-nnet-arch.png'):
        file_path = os.path.join(folder, filename)
        visualkeras.layered_view(self.nnet.model, legend=True, to_file=file_path, spacing=5)
