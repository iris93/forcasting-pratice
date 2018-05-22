from __future__ import absolute_import
from __future__ import division

import os
import sys
import ConfigParser
from log import logger
from CNN_config_parser import basedir, init_preprocessing

config = ConfigParser.ConfigParser()
config_file = os.path.join(basedir, 'CNN_config.ini')
# Loading module config_parser to obtain configuration parameters
development, yahoo_excel = None, None
yahoo_historical_train_from, yahoo_historical_train_to = None, None
yahoo_historical_test_from, yahoo_historical_test_to = None, None
try:
	config.readfp(open(config_file))
	IMAGE_SIZE,NUM_CHANNELS,PIXEL_DEPTH,NUM_LABELS,VALIDATION_SIZE,SEED,NUM_EPOCHS,BATCH_SIZE,EVAL_BATCH_SIZE,EVAL_FREQUENCY,iteration,fc1_shape0,fc1_shape1,learning_rate_ini,decay_rate= init_preprocessing(config)
except:
	logger.debug('Error init_preprocessing(). Exit!')
	sys.exit()
print "          IMAGE_SIZE=",IMAGE_SIZE,";        NUM_CHANNELS=",NUM_CHANNELS,";           PIXEL_DEPTH=",PIXEL_DEPTH,";               NUM_LABELS=",NUM_LABELS
print "VALIDATION_SIZE=",VALIDATION_SIZE,";     SEED=",SEED,";                         NUM_EPOCHS=",NUM_EPOCHS,";        BATCH_SIZE=",BATCH_SIZE
print "EVAL_BATCH_SIZE=",EVAL_BATCH_SIZE,";     EVAL_FREQUENCY=",EVAL_FREQUENCY,";     iteration=",iteration
print "            fc1_shape0=",fc1_shape0,";   fc1_shape1=",fc1_shape1,";                  learning_rate_ini=",learning_rate_ini,";     decay_rate=",decay_rate