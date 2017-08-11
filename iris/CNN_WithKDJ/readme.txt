How to use:

Python requirement:
- configparser
- numpy
- pandas
- tensorflow
- yahoo_finance

1. CNN_inputdata/CNN_Input_WithKDJ.py
	you can adjust the data you got from yahoo finace to the form you like.and save the data to teX.npy,teY.npy,trX.npy,trY.npy to be prepared for using.

2.CNN_inputdata/config.ini
	you can adjust the dates to get the data that you want between the dates.
	but you have to run CNN_Input_WithKDJ.py after you change config.ini

3.CNN/CNN_WithKDJ.py
	this is the core of the CNN model,just run it to train the model and see the result.
	you can adjust the parameter in CNN_config.ini.