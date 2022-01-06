import os
import logging
import coloredlogs
import time

LOG_DIR_PATH = os.path.join('logs')


def init_logger():
    # Check log dir
    check_log_dir()
    # Init dir
    log_file = os.path.join(LOG_DIR_PATH, '{}.log'.format(time.strftime("%d_%m_%Y-%H_%M_%S")))
    logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s : %(name)s : %(levelname)s >>> %(message)s')
    coloredlogs.install(level='INFO')


def check_log_dir():
    os.makedirs(LOG_DIR_PATH, exist_ok=True)
