'''
Instituto de Ciencias Matematicas e de Computacao - USP São Carlos
SCC5809: Redes Neurais

Projeto 01: MLP - Backpropagation
Equipe:
ID. Matricula (01) - 12116252 Dheniffer Caroline Araújo Pessoa

ID. Matricula (02) - 12114819 Douglas Queiroz Galucio Batista 

ID. Matricula (03) - 12116738 Laleska Mesquita
'''


# Importação das bibliotecas
import numpy as np
import os
import random
from prettytable import PrettyTable
from sklearn.preprocessing import scale

# Classe que representa uma MLP (Perceptron Multi-Camada)
class MLP(object):

	# Dicionário do python para armazenar o modelo da MLP
	model = {}
	dE2_dw_h_b = 0
	dE2_dw_o_b = 0

	# Construtor da classe
	def __init__(self):
		# Inicializa os parâmetros para a MLP
		#self.architecture()
		return

	# Função de Ativação
	def fnet(net):
		return (1 / (1 + np.exp(-net)))

	# Função para calcular a derivada
	def df_dnet(f_net):
		return (f_net * (1 - f_net))

	# Função que inicializa a arquitetura da MLP 
	def architecture(self,input_lenght=13,hidden_lenght=5,output_lenght=3,fnet=fnet,df_dnet=df_dnet):
		self.model['input_lenght'] = input_lenght
		self.model['hidden_lenght'] = hidden_lenght
		self.model['output_lenght'] = output_lenght
		self.model['hidden_layer'] = np.random.uniform(-0.5,+0.5,(hidden_lenght,input_lenght+1))
		self.model['output_layer'] = np.random.uniform(-0.5,+0.5,(output_lenght,hidden_lenght+1))
		self.model['fnet'] = fnet
		self.model['df_dnet'] = df_dnet

	# Função que efetua o forward do algoritmo, no qual é calculado o output final a partir dos pesos atuais
	def forward(self, X):
		#Removendo os valores do modelo
		hidden = self.model['hidden_layer'] 
		output = self.model['output_layer']
		#Adicionando o 1 para a multiplicação
		X = np.concatenate((X,np.array([1])))

		
		#Camada Oculta
		net_h = np.matmul(hidden,X)
		#print('net_h=',net_h,'\n')
		f_net_h = self.model['fnet'](net_h)
		#print('f_net_h=',f_net_h,'\n')
		df_net_h = self.model['df_dnet'](f_net_h)
		#f_net_h = np.rint(f_net_h)
		
		#Camada de saída
		f_net_h_c = np.concatenate((f_net_h,np.array([1])))
		net_o = np.matmul(output,f_net_h_c)
		#print('net_o=',net_o,'\n')
		#print('net_o=',net_o,'\n')
		f_net_o = self.model['fnet'](net_o)
		#print('f_net_o=',f_net_o,'\n')
		df_net_o = self.model['df_dnet'](f_net_o)
		#f_net_o = np.rint(f_net_o)

		return{
			"f_net_h": f_net_h,
			"df_net_h":df_net_h,
			"f_net_o": f_net_o,
			"df_net_o":df_net_o
		}

	# Função que realiza o treinamento da rede utilizando o backpropagation com regra delta
	def backpropagation(self,X,Y,eta=0.5,alpha=0.5,max_error=0.000001,max_iter=500):
		counter = 0
		total_error = 2*max_error

		# Estrutura de repetição do treinamento que acontece enquanto o erro for maior que o aceitável ou o numero máximo de iterações não tiver sido atingido
		while total_error > max_error and counter < max_iter:
			total_error = 0

			for i in range(0,X.shape[0]):
				x_i = X[i,:]
				y_i = Y[i]
				#print('x_i=',x_i,'\n')
				#print('y_i=',y_i,'\n')

				#forward
				fw = self.forward(x_i)

				#erro
				error_o_k = (y_i-fw['f_net_o'])
				#print('y_i',y_i,'\n')
				#print('f_net_o',fw['f_net_o'],'\n')
				#print('error_o_k',error_o_k,'\n')
				
				total_error = total_error + np.sum(error_o_k*error_o_k)

				#backpropagation / cálculo das derivadas
				delta_out = error_o_k*fw['df_net_o']
				dE2_dw_o = np.multiply(np.array([-2*delta_out]).T,np.concatenate((fw['f_net_h'],np.array([1]))))

				delta_h = np.matmul(np.array([delta_out]),self.model['output_layer'][:,0:self.model['hidden_lenght']])
				dE2_dw_h = delta_h * (np.multiply(-2*fw['df_net_h'],np.array([np.concatenate((x_i,np.array([1])))]).T))


				# Atualização dos pesos
				self.model['output_layer'] = self.model['output_layer'] - eta*dE2_dw_o - alpha*self.dE2_dw_o_b
				if counter == 0:
					self.model['hidden_layer'] = self.model['hidden_layer'] - np.reshape(np.array([eta*dE2_dw_h]).T,(self.model['hidden_lenght'],x_i.size+1))
				else:
					self.model['hidden_layer'] = self.model['hidden_layer'] - np.reshape(np.array([eta*dE2_dw_h]).T,(self.model['hidden_lenght'],x_i.size+1)) - np.reshape(np.array([alpha*self.dE2_dw_h_b]).T,(self.model['hidden_lenght'],x_i.size+1))
				

				self.dE2_dw_o_b = dE2_dw_o
				self.dE2_dw_h_b = dE2_dw_h

			# Término da iteração da função de treinamento
			total_error = total_error/X.shape[0]
			counter = counter+1
			if (counter % 100) == 0:
				print("Iteração:",counter," Error:",total_error)

		return

	def run(self,X,Y,task,size=8,eta=0.1,alpha=0.5,max_iter=500,train_size=0.7,threshold=0.000001):
		ids = random.sample(range(0,X.shape[0]),np.floor(train_size*X.shape[0]).astype(np.int))
		ids_left = diff(range(0,X.shape[0]),ids)
		#print('ids',ids,'\n')
		#print('ids_left',ids_left,'\n')

		# Treinando Set
		train_set = X[ids,:]
		train_classes = Y[ids,:]
		#print('X=',train_set)
		#print('Y=',train_classes)

		# Testando Set
		test_set = X[ids_left,:]
		test_classes = Y[ids_left,:]
		#print('X=',test_set)
		#print('Y=',test_classes)

		self.architecture(input_lenght=X.shape[1],hidden_lenght=size,output_lenght=Y.shape[1])
		print('Arquitetura da MLP Criada\nComeçando o treino')
		self.backpropagation(train_set,train_classes,eta=eta,alpha=alpha,max_error=threshold,max_iter=max_iter)
		print('Rede Neural Treinada\nComeçando o teste')
		#print(mlp.forward(X)['f_net_o'])

		correct = 0
		sqerror = 0
		for i in range(0,test_set.shape[0]):
			x_i = test_set[i]
			y_i = test_classes[i]

			if task == 'C':
				y_hat_i = np.round(self.forward(x_i)['f_net_o'])
			if task == 'R':
				y_hat_i = self.forward(x_i)['f_net_o']
			
			error = y_i - y_hat_i
			if (np.sum((error)**2) == 0):
				correct = correct + 1
				print('correct: ', correct)
			sqerror = sqerror + np.sum(error*error)
			#print('\n>>>>>>>>>>CORRECT: ', correct)

			pass

		print('Rede Neural Testada')
		accuracy = correct/test_set.shape[0]
		sqerror = sqerror/test_set.shape[0]
		
		return {
			"precisao": accuracy,
			"err": sqerror
		}
	#Fim da classe MLP

#Diferença
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]