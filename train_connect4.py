import logging

import coloredlogs
import os

from keras.utils.vis_utils import plot_model

from src.Coach import Coach
from src.connect4.Connect4Game import Connect4Game
from src.connect4.keras.NNet import NNetWrapper

from multiprocessing import cpu_count


def init_log(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    logging.basicConfig(filename=filename, filemode='a')
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
    return logging.getLogger(__name__)


log = init_log('run_unfo.log')

args = dict(
    numIters=1000,
    numEps=100,              # Number of complete self-play games to simulate during a new iteration.
    tempThreshold=15,        #
    updateThreshold=0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    maxlenOfQueue=100000,    # Number of game examples to train the neural networks.
    numMCTSSims=25,          # Number of games moves for MCTS to simulate.
    arenaCompare=10,         # Number of games to play during arena play to determine if new net will be accepted.
    cpuct=1,

    checkpoint='temp/',
    load_model=False,
    load_folder_file=('/dev/models/8x100x50', 'best.pth.tar'),
    numItersForTrainExamplesHistory=20,
    nThreads=cpu_count()
)


def main():
    log.info('Loading %s...', Connect4Game.__name__)
    game = Connect4Game()

    log.info('Loading %s...', NNetWrapper.__name__)
    nnet = NNetWrapper(game)

    # Neural network info
    log.info('Neural network info:')
    log.info(nnet.nnet.model.summary())
    #plot_model(nnet.nnet.model, to_file=os.path.join('./src', 'connect4', 'keras', 'model.png'))

    # Load weigths
    if args['load_model']:
        log.info('Loading checkpoint "%s/%s"...', args['load_folder_file'][0], args['load_folder_file'][1])
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    else:
        log.warning('Not loading a checkpoint!')

    # Create coach
    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    if args['load_model']:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    # Start training
    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
