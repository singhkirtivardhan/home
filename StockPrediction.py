
import numpy as np
import pandas as pd
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import pickle
import StockPrice


# This function saves the company stock data in a file
def save_data_file(company_name):
	cols = ['ticker', 'date', 'close']
	prices_dataset =  pd.read_csv('../FP/WIKI.csv', usecols=cols)

	data = prices_dataset[prices_dataset['ticker']==company_name]
	print company_name, ' data size = ', len(data.close.values)
	pickle.dump(data, open(company_name+'.npy', 'w'))


# This function converts the array of prices into dataset where x contains
# the stock price at time t, t-1, t-2, ..., t-steps+1 and y contains
# the stock price at time t+1
def create_dataset(dataset, steps):
	dataX, dataY = [], []
	for i in range(len(dataset)-steps-1):
		a = dataset[i:(i+steps), 0]
		dataX.append(a)
		dataY.append(dataset[i+steps, 0])
	return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
	'''# use this part to generate the data file for specific companies
	companies = ['AMZN', 'GOOGL']
	for i in companies:
		save_data_file(i)'''

	# different values for hyperparameters
	steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	batch_size = [16, 32, 64, 128, 256, 512]
	dropout = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2]

	for s in steps:
		for bs in batch_size:
			for d in dropout:
				print "M1 - GOOGL --------- bs = ", bs, " - steps = ", s, " - dr = ", d
				sp = StockPrice('GOOGL', s, bs, d)
				#sp.load_model_from_file('./GOOGL_model-bs'+str(bs)+'-steps'+str(s)+'-dr'+str(d))
				sp.train()
				sp.evaluate()
				#sp.plot_results()
