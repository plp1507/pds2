import numpy as np
import matplotlib.pyplot as plt

#### Entradas e saídas desejadas
X = np.asarray([[0,0],[0,1],[1,0],[1,1]])

Y_and = np.asarray([0,0,0,1])
Y_or  = np.asarray([0,1,1,1])

#### Inicializações
N = np.shape(X)[1]
n_padroes = np.shape(X)[0]

X = np.hstack((X, -1*np.ones([n_padroes, 1]))) # inserção do bias nas entradas

W = np.random.randn(N+1) # pesos com bias (N+1 elemento)

eta = 100000      # taxa de aprendizado
tolerancia = 0.001  #tolerância para o erro quadrático médio por época
erro = 0
vetor_erros = []


