import logging
import os
import sys
import time

from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

from src.connect4.Connect4Players import MCTSConnect4Player

from multiprocessing import Pool

from tqdm import tqdm

import numpy as np

from src.Arena import Arena
from src.MCTS import MCTS

log = logging.getLogger(__name__)


def compute_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), round(seconds))


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False
        # History of pitting
        self.pitting_history = []

    @staticmethod
    def executeEpisode(mcts: MCTS):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = mcts.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = mcts.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < mcts.args['tempThreshold'])

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = mcts.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = mcts.game.getNextState(board, curPlayer, action)

            r = mcts.game.getGameEnded(board, curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        starting_time = time.time()

        for i in range(1, self.args['numIters'] + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ... (Elapsed time: {compute_elapsed_time(starting_time, time.time())})')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args['maxlenOfQueue'])

                with Pool(self.args['nThreads']) as pool:
                    episodes = [MCTS(self.game, self.nnet, dict(self.args)) for _ in range(self.args['numEps'])]
                    train_examples = list(tqdm(pool.imap(self.executeEpisode, episodes), total=self.args['numEps']))

                # Add train example
                for example in train_examples:
                    iterationTrainExamples += example

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args['numItersForTrainExamplesHistory']:
                log.warning(
                    f"Removing the oldest entry in trainExamples. "
                    f"len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.h5')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.h5')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Create arena
            log.info('PITTING AGAINST PREVIOUS VERSION')
            p1, p2 = MCTSConnect4Player(self.game, 1, pmcts), MCTSConnect4Player(self.game, -1, nmcts)
            arena = Arena(p1.play, p2.play, self.game)

            # Play multiple games
            pwins, nwins, draws = arena.playGames(self.args['arenaCompare'])
            self.pitting_history.append((pwins, nwins, draws))

            # Print games info
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args['updateThreshold']:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.h5')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.h5')

    @staticmethod
    def getCheckpointFile(iteration):
        return 'checkpoint_' + str(iteration) + '.h5'

    def saveTrainExamples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
