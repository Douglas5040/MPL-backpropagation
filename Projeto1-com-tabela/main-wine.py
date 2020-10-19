
from mlp import MLP
import numpy as np
from prettytable import PrettyTable
import os

# Reads the contents from a file and transforms in matrix
def matrix(contents):
	return [item.split(',') for item in contents.split('\n')[:-1]]

def class_ind(Y):
	unique_Y = set(Y)
	#print('unique_Y=',len(unique_Y),'\n')
	size = (Y.shape[0],len(unique_Y))
	res = np.zeros(size)
	for i in range(0,Y.shape[0]):
		res[i][Y[i].astype(np.int)-1] = 1

	#print('res=',res,'\n')
	return res

def wine_test(eta=0.1,alpha=0,max_iter=500,train_size=0.7):
	for file in os.listdir():
		if(file.endswith('.data')):
			# Preprocessing wine data set
			data = open(file).read()
			X = matrix(data)
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:,0]
			X = X[:,1:X.shape[1]]
			# Normalizing X
			for i in range(X.shape[1]):
				X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))
			
			# Binarizing the classes output
			Y = class_ind(Y)
			

	print('\nPreprocessing Wine Done')
	mlp = MLP()
	return mlp.run(X,Y,'C',eta=eta,alpha=alpha,max_iter=max_iter,train_size=train_size)


print('Wine')
table = PrettyTable()
table.field_names = ["Number of Cycles","Learning Speed","Momentum","Training set size","Square Error","Accuracy"]

# Variation Learning Speed
ret = wine_test(eta=0.1)
table.add_row([500,0.1,0,0.7,ret['error'],ret['accuracy']])
ret = wine_test(eta=0.3)
table.add_row([500,0.3,0,0.7,ret['error'],ret['accuracy']])
ret = wine_test(eta=0.5)
table.add_row([500,0.5,0,0.7,ret['error'],ret['accuracy']])

# Variating Number of Cycles
ret = wine_test(max_iter=250)
table.add_row([250,0.1,0,0.7,ret['error'],ret['accuracy']])
ret = wine_test(max_iter=750)
table.add_row([750,0.1,0,0.7,ret['error'],ret['accuracy']])
ret = wine_test(max_iter=1000)
table.add_row([1000,0.1,0,0.7,ret['error'],ret['accuracy']])

# Variating Training set size
ret = wine_test(train_size=0.5)
table.add_row([500,0.1,0,0.5,ret['error'],ret['accuracy']])
ret = wine_test(train_size=0.6)
table.add_row([500,0.1,0,0.6,ret['error'],ret['accuracy']])
ret = wine_test(train_size=0.9)
table.add_row([500,0.1,0,0.9,ret['error'],ret['accuracy']])

# Variation Momentum
ret = wine_test(alpha=0.1)
table.add_row([500,0.1,0.1,0.7,ret['error'],ret['accuracy']])
ret = wine_test(alpha=0.3)
table.add_row([500,0.1,0.3,0.7,ret['error'],ret['accuracy']])
ret = wine_test(alpha=0.7)
table.add_row([500,0.1,0.7,0.7,ret['error'],ret['accuracy']])

# Printing Results
print(table)
