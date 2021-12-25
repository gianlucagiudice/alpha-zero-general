from src.Game import Game

from keras.layers import Input, Reshape, Activation, BatchNormalization, Conv2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizer_v2.adam import Adam
import tensorflow as tf
tf.get_logger().setLevel('WARNING')


class Connect4NNet:
    def __init__(self, game: Game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))

        # Parameters
        num_channels = args["num_channels"]
        dropout = args["dropout"]
        lr = args["lr"]

        # ----- Define network architecture -----

        # Input
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)

        # Hidden convolutional layers
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(h_conv1)))
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='same')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='valid')(h_conv3)))

        # Flatten layers after convolutions
        h_conv4_flat = Flatten()(h_conv4)

        # Dense layers
        s_fc1 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))
        s_fc2 = Dropout(dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))

        # Multitask learning
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        # Create model and compile
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(lr))
