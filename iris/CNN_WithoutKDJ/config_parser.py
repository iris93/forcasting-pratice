# coding: utf-8
# The following file is reading config.ini
import os.path

basedir = os.path.abspath(os.path.dirname(__file__))

# initialization for preprocessing
def init_preprocessing(config):
	try:
		# Development or Production
		development = config.get('preprocessing', 'development')

		# Test measurement by cross validation
		cross_validation = config.get('preprocessing', 'cross_validation')

		# Excel file for development purpose
		yahoo_excel = config.get('preprocessing', 'yahoo_excel')
		yahoo_excel = os.path.join(basedir, 'reference_code', yahoo_excel)

		# yahoo historical data for training
		yahoo_historical_train_from = config.get('preprocessing', 'yahoo_historical_train_from')
		yahoo_historical_train_to = config.get('preprocessing', 'yahoo_historical_train_to')

		# yahoo historical data for test
		yahoo_historical_test_from = config.get('preprocessing', 'yahoo_historical_test_from')
		yahoo_historical_test_to = config.get('preprocessing', 'yahoo_historical_test_to')
		
	except:
		return None, None, None, None, None, None, None

	return development, cross_validation, yahoo_excel, yahoo_historical_train_from, yahoo_historical_train_to, yahoo_historical_test_from, yahoo_historical_test_to
