# coding: utf-8
# The following file is reading CNN_config.ini
import os.path

basedir = os.path.abspath(os.path.dirname(__file__))

# initialization for preprocessing
def init_preprocessing(config):
	try:
		# get image size 
		IMAGE_SIZE = config.get('preprocessing', 'IMAGE_SIZE')

		# 
		NUM_CHANNELS =config.get('preprocessing', 'NUM_CHANNELS')

		# 
		PIXEL_DEPTH = config.get('preprocessing', 'PIXEL_DEPTH')

		# 
		NUM_LABELS = config.get('preprocessing', 'NUM_LABELS')

		# 
		VALIDATION_SIZE = config.get('preprocessing', 'VALIDATION_SIZE')

		# 
		SEED = config.get('preprocessing', 'SEED')

		# 
		BATCH_SIZE = config.get('preprocessing', 'BATCH_SIZE')

		# 
		NUM_EPOCHS = config.get('preprocessing', 'NUM_EPOCHS')


		# 
		EVAL_BATCH_SIZE = config.get('preprocessing', 'EVAL_BATCH_SIZE')

		# 
		EVAL_FREQUENCY = config.get('preprocessing', 'EVAL_FREQUENCY')
		# 
		iteration = config.get('preprocessing', 'iteration')
		# 		
		# 
		fc1_shape0 = config.get('preprocessing', 'fc1_shape0')
		# 
		fc1_shape1 = config.get('preprocessing', 'fc1_shape1')

		# 
		learning_rate_ini = config.get('preprocessing', 'learning_rate_ini')

		# 
		decay_rate = config.get('preprocessing', 'decay_rate')
	except:
		return None, None, None, None, None, None

	return int(IMAGE_SIZE),int(NUM_CHANNELS),int(PIXEL_DEPTH),int(NUM_LABELS),int(VALIDATION_SIZE),int(SEED),int(NUM_EPOCHS),int(BATCH_SIZE),int(EVAL_BATCH_SIZE),int(EVAL_FREQUENCY),int(iteration),int(fc1_shape0),int(fc1_shape1),float(learning_rate_ini),float(decay_rate)
