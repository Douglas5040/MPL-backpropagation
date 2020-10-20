
from mlp import MLP
import numpy as np
from prettytable import PrettyTable
import os

# Reads the contents from a file and transforms in matrix


def matrix(contents):
	return [item.split(',') for item in contents.split('\n')[:-1]]

def tracks_test(eta=0.1,alpha=0.5,max_iter=500,train_size=0.7):
	mlp = MLP()
	n = 1
	ret = {}
	for file in os.listdir("datasets"):
		if(file.endswith('.txt')):
			print('file: ', file)
			# Pré-processamento dos arquivos do Dataset 'Origin of Music'
			data = open('datasets/'+file).read()
			X = matrix(data)
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:,X.shape[1]-2:X.shape[1]]
			X = X[:,0:X.shape[1]-2]

			# Normalizando X e Y
			for i in range(X.shape[1]):
				X[:,i] = (X[:,i] - np.amin(X[:,i])) / (np.amax(X[:,i]) - np.amin(X[:,i]))
			for i in range(Y.shape[1]):
				Y[:,i] = (Y[:,i] - np.amin(Y[:,i])) / (np.amax(Y[:,i]) - np.amin(Y[:,i]))
			
			print('\nPreprocessing Origin of Music file ',n,' Done')
			res = mlp.run(X,Y,'R',eta=eta,alpha=alpha,max_iter=max_iter,train_size=train_size)
			ret['accuracy'] = res['accuracy']
			

			# Salvando o erro para cada arquivo em posições diferentes
			if n == 1:
				ret['error_1'] = res['error']
			if n == 2:
				ret['error_2'] = res['error']

			# Adicionando o contador do arquivo
			n = n+1

	return ret

print('Origin of Music Choosen')
table = PrettyTable()
table.field_names = ["Number of Cycles","Learning Speed","Momentum","Training set size","First file Mean Square Error","Second file Mean Square Error", "Accuracy"]

# Variating Learning Speed
err = tracks_test(eta=0.1)
table.add_row([500,0.1,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.3)
table.add_row([500,0.3,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5)
table.add_row([500,0.5,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])

# Variating Number of Cycles
err = tracks_test(eta=0.5,max_iter=300)
table.add_row([300,0.5,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,max_iter=500)
table.add_row([700,0.5,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,max_iter=700)
table.add_row([1000,0.5,0.5,0.7,err['error_1'],err['error_2'],err['accuracy']])

# Variating Training Set Size
err = tracks_test(eta=0.5,train_size=0.5)
table.add_row([500,0.5,0.5,0.5,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,train_size=0.75)
table.add_row([500,0.5,0.5,0.75,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,train_size=0.9)
table.add_row([500,0.5,0.5,0.9,err['error_1'],err['error_2'],err['accuracy']])

# Variating Momentum
err = tracks_test(eta=0.5,alpha=0.1)
table.add_row([500,0.5,0.1,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,alpha=0.3)
table.add_row([500,0.5,0.3,0.7,err['error_1'],err['error_2'],err['accuracy']])
err = tracks_test(eta=0.5,alpha=0.7)
table.add_row([500,0.5,0.7,0.7,err['error_1'],err['error_2'],err['accuracy']])

# Printing Results
print(table)
