# coding: utf-8
import os
import time
import logging
from   logging.handlers import TimedRotatingFileHandler

basedir = os.path.abspath(os.path.dirname(__file__))
log_path = os.path.join(basedir, 'log', 'debug')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(asctime)s] %(message)s', '%Y-%m-%d %H:%M:%S')

filehandler = TimedRotatingFileHandler(log_path, 'D', 1, 0)
filehandler.setFormatter(formatter)

filehandler.suffix = '%Y-%m-%d_%H:%M:%S'
logger.addHandler(filehandler)
