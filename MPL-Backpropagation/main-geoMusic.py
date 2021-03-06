'''
Instituto de Ciencias Matematicas e de Computacao - USP São Carlos
SCC5809: Redes Neurais

Projeto 01: MLP - Backpropagation
Equipe:
ID. Matricula (01) - 12116252 Dheniffer Caroline Araújo Pessoa

ID. Matricula (02) - 12114819 Douglas Queiroz Galucio Batista 

ID. Matricula (03) - 12116738 Laleska Mesquita
'''
import numpy as np
import os
from mlp import MLP
from prettytable import PrettyTable


# Método para ler o conteúdo e transformar em matriz

def dados_in(contents): #matrix
	return [item.split(',') for item in contents.split('\n')[:-1]]

# Aqui será definido os parametros de treinamento 
def music_geo_test(eta=0.1, alpha=0.5, max_iter=500, train_size=0.7): #track_testes
	mlp = MLP()

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


	print('Processando do Music ')
	return mlp.run(X, Y, 'R', alpha=alpha, max_iter=max_iter, eta=eta, train_size=train_size)

print('Music')
table = PrettyTable()
table.field_names = ["Numero de Ciclos","Velocidade de Aprendizagem","Momento","Tamanho do conj Treinamento","Erro do quadrado médio"]

# Velocidade de Aprendizagem
print('\n\n### Testando Velocidade de Aprendizagem ###')
err = music_geo_test(eta=0.1)
table.add_row([500,0.1,0.5,0.7,err['err']])
err = music_geo_test(eta=0.3)
table.add_row([500,0.3,0.5,0.7,err['err']])
err = music_geo_test(eta=0.5)
table.add_row([500,0.5,0.5,0.7,err['err']])

# Número de Ciclos
print('\n\n### Testando Número de Ciclos ###')
err = music_geo_test(eta=0.5,max_iter=300)
table.add_row([300,0.5,0.5,0.7,err['err']])
err = music_geo_test(eta=0.5,max_iter=500)
table.add_row([700,0.5,0.5,0.7,err['err']])
err = music_geo_test(eta=0.5,max_iter=700)
table.add_row([1000,0.5,0.5,0.7,err['err']])
err = music_geo_test(eta=0.5,max_iter=1000)
table.add_row([1300,0.5,0.5,0.7,err['err']])

# Conjunto de Treinamento 
print('\n\n### Testando Conjunto de Treinamento ###')
err = music_geo_test(eta=0.5,train_size=0.5)
table.add_row([500,0.5,0.5,0.5,err['err']])
err = music_geo_test(eta=0.5,train_size=0.75)
table.add_row([500,0.5,0.5,0.75,err['err']])
err = music_geo_test(eta=0.5,train_size=0.8)
table.add_row([500,0.5,0.5,0.9,err['err']])
err = music_geo_test(eta=0.5,train_size=0.9)
table.add_row([500,0.5,0.5,0.9,err['err']])

# Momentum
print('\n\n### Testando Momentum ###')
err = music_geo_test(eta=0.5,alpha=0.1)
table.add_row([500,0.5,0.1,0.7,err['err']])
err = music_geo_test(eta=0.5,alpha=0.3)
table.add_row([500,0.5,0.3,0.7,err['err']])
err = music_geo_test(eta=0.5,alpha=0.7)
table.add_row([500,0.5,0.7,0.7,err['err']])
err = music_geo_test(eta=0.5,alpha=1)
table.add_row([500,0.5,0,1,err['err']])

# Imprime os resultados na tabela
print(table)