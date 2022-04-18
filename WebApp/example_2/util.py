from sklearn.svm import SVR
import numpy as np
import pandas as pd
import pickle



def save_model(filename = 'data/hw_regression_data.csv', model_name = 'model/regr.sav'):
	'''
	This function saves model in the model folder
	By default, use study hours to predict final grades
	'''
	df = pd.read_csv(filename)

	# first col is y, rest cols are X
	X = df.values[:,1:]
	y = df.values[:,0]

	regr = SVR(C=1.0, epsilon=0.2)
	regr.fit(X, y)

	pickle.dump(regr, open(model_name, 'wb'))


def load_model_pred(filename = 'data/test.csv', model_name = 'model/regr.sav'):
	'''
	This function load the model and output predictions
	By default, use study hours to predict final grades
	'''
	df = pd.read_csv(filename)

	# first col is y, rest cols are X
	# TODO
	# TODO Load model
	return '''TODO return predictions of the test data'''