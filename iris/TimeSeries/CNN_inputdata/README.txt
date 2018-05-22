How to use:

Python requirement:
- configparser
- numpy
- pandas
- tensorflow
- yahoo_finance


1. config.ini
   [preprocessing]
   development = 0 # for data source offline excel (only contain input 4 stock prices and 1 volume)
   development = 1 # crawling data from yahoo stock finance and have option kdj='yes' or kdj='no', will be explained later.

   yahoo_historical_train_from    is train data started from
   yahoo_historical_train_to      is train data until

   yahoo_historical_test_from     is test data started from
   yahoo_historical_test_to       is test data until


2. preparing_input.py

   # import module
   from preparing_input import InitPreprocessing

   # create a new instance
   init_instance = InitPreprocessing()

   # -----------------------------------
   # get_stock(), get real stock data
   # kdj='no', for input without KDJ otherwise set kdj='yes'
   # kdj only support for production setting, see config.ini

   # kdj_style='google', automatically kdj_short_period=3 and kdj_long_period=5
   #    for example: output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5

   # kdj_style='', need to set kdj_short_period and kdj_long_period
   #    for example: output='close', kdj='yes', kdj_style='', kdj_short_period=3, kdj_long_period=6
   X_train, y_train, X_test, y_test = init_instance.get_stock(output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5)

   # -----------------------------------
   # get_stock_transform(), get transform stock data
   # kdj='no', for input without KDJ otherwise set kdj='yes'
   # kdj only support for production setting, see config.ini

   # kdj_style='google', automatically kdj_short_period=3 and kdj_long_period=5
   #    for example: output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5

   # kdj_style='', need to set kdj_short_period and kdj_long_period
   #    for example: output='close', kdj='yes', kdj_style='', kdj_short_period=3, kdj_long_period=6
   X_train, y_train, X_test, y_test = init_instance.get_stock_transform(output='close', kdj='yes', kdj_style='google', kdj_short_period=3, kdj_long_period=5)

   # -----------------------------------
   # get_X_train_max(), get maximum value
   # get_X_test_max(), get maximum value
   kdj='yes'   # for production with kdj
   X_train_max_open, X_train_max_high, X_train_max_low, X_train_max_close, X_train_max_volume, X_train_max_K, X_train_max_D, X_train_max_J = init_instance.get_X_train_max(kdj=kdj)
   X_test_max_open, X_test_max_high, X_test_max_low, X_test_max_close, X_test_max_volume, X_test_max_K, X_test_max_D, X_test_max_J = init_instance.get_X_test_max(kdj=kdj)
   y_train_max_open, y_train_max_high, y_train_max_low, y_train_max_close = init_instance.get_y_train_max()
   y_test_max_open, y_test_max_high, y_test_max_low, y_test_max_close = init_instance.get_y_test_max()


3. nn_sample.py
   Sample how to use preparing_input.py
