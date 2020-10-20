import numpy as np
import os
from mlp import MLP
from prettytable import PrettyTable


# Aqui vamos ler o conteudo e tranformar em matriz

def dados_in(contents): #matrix
	return [item.split(',') for item in contents.split('\n')[:-1]]

# Aqui será definido os parametros de treinamento 
def music_geo_test(eta=0.1, alpha=0.5, max_iter=500, train_size=0.7): #track_testes
	mlp = MLP()
	n = 1
	ret = {}
	for file in os.listdir("datasets"):
		if(file.endswith('.txt')):
			print('\nfile: ', file)
			# Aqui vamos fazer um processamento inicial do aquivo Music
			data = open('datasets/'+file).read()
			X = dados_in(data)   #matrix
			X = np.array(X)
			X = X.astype(np.float)
			Y = X[:, X.shape[1]-2:X.shape[1]]
			X = X[:, 0:X.shape[1]-2]

			for i in range(X.shape[1]):
				X[:, i] = (X[:, i] - np.amin(X[:, i])) / \
				           (np.amax(X[:, i]) - np.amin(X[:, i]))
			for i in range(Y.shape[1]):
				Y[:, i] = (Y[:, i] - np.amin(Y[:, i])) / \
				           (np.amax(Y[:, i]) - np.amin(Y[:, i]))

			print('Processando o aquivo Music ', n, ' okay')
			res = mlp.run(X, Y, 'R', alpha=alpha, max_iter=max_iter, eta=eta, train_size=train_size)
			ret['precisao'] = res['precisao']

			# armazenando valores da variavel erro
			if (n == 1):
				ret['err_1'] = res['err']
			if n == 2:
				ret['err_2'] = res['err']

			# Contador
			n = n+1

	return ret

print('Music')
table = PrettyTable()
table.field_names = ["Numero de Ciclos","Velocidade de Aprendizagem","Momento","Tamanho do conj Treinamento","Erro do quadrado médio do primeiro arquivo","erro quadratico médio segundo arquivo", "precisao"]

# Velocidade de Aprendizagem
print('\n\n### Testando Velocidade de Aprendizagem ###')
err = music_geo_test(eta=0.1)
table.add_row([500,0.1,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.3)
table.add_row([500,0.3,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5)
table.add_row([500,0.5,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])

# Numero de Ciclos
print('\n\n### Testando Numero de Ciclos ###')
err = music_geo_test(eta=0.5,max_iter=300)
table.add_row([300,0.5,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,max_iter=500)
table.add_row([700,0.5,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,max_iter=700)
table.add_row([1000,0.5,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,max_iter=1000)
table.add_row([1300,0.5,0.5,0.7,err['err_1'],err['err_2'],err['precisao']])

# Conjunto de Treinamento 
print('\n\n### Testando Conjunto de Treinamento ###')
err = music_geo_test(eta=0.5,train_size=0.5)
table.add_row([500,0.5,0.5,0.5,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,train_size=0.75)
table.add_row([500,0.5,0.5,0.75,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,train_size=0.9)
table.add_row([500,0.5,0.5,0.9,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,train_size=1.5)
table.add_row([500,0.5,0.5,0.9,err['err_1'],err['err_2'],err['precisao']])

# Momento 
print('\n\n### Testando Momento ###')
err = music_geo_test(eta=0.5,alpha=0.1)
table.add_row([500,0.5,0.1,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,alpha=0.3)
table.add_row([500,0.5,0.3,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,alpha=0.7)
table.add_row([500,0.5,0.7,0.7,err['err_1'],err['err_2'],err['precisao']])
err = music_geo_test(eta=0.5,alpha=1)
table.add_row([500,0.5,0,0.7,err['err_1'],err['err_2'],err['precisao']])

# Mostra tabela
print(table)