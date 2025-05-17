# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:00:15 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D  #gráficos 3D

porta = int(input("Digite 1 para treinar a porta NAND de 3 entradas, ou 2 para NOR: "))

X = np.array(list(product([0, 1], repeat=3)))  #entradas possíveis com 3 bits, 8 combinações

if porta == 1:
    # NAND de 3 entradas: só 0 quando todas forem 1
    Yd = np.array([0 if np.all(x == 1) else 1 for x in X])
    titulo = "Porta NAND - 3 Entradas"
elif porta == 2:
    # NOR de 3 entradas: só 1 quando todas forem 0
    Yd = np.array([1 if np.all(x == 0) else 0 for x in X])
    titulo = "Porta NOR - 3 Entradas"


NumeroPadroes = X.shape[0] #adicionando bias às entradas (-1)
X_bias = -1 * np.ones((NumeroPadroes, 1))
X = np.hstack((X, X_bias))  #entradas com bias

W = np.random.randn(1, X.shape[1]) #pesos aleatórios

#parâmetros do treinamento
Eta = 0.1               # taxa de aprendizado
Tolerancia = 0.001      # critério de parada
Vetor_Erros = []        # lista para armazenar o erro por época

#loop de treinamento
while True:
    ErroEpoca = 0
    indices = np.random.permutation(NumeroPadroes)

    for i in indices:
        xi = X[i]
        yd = Yd[i]

        y = int(np.dot(W, xi.T) >= 0)
        erro = yd - y

        #atualização dos pesos
        W += Eta * erro * xi

        ErroEpoca += erro ** 2

    Vetor_Erros.append(ErroEpoca)

    if ErroEpoca < Tolerancia:
        break


print("\nPesos finais:")
print(f"  Peso 1: {W[0,0]:.4f}")
print(f"  Peso 2: {W[0,1]:.4f}")
print(f"  Peso 3: {W[0,2]:.4f}")
print(f"  Bias  : {W[0,3]:.4f}")

#plot erro quadrático por época
plt.plot(Vetor_Erros, marker='o')
plt.title(f"Evolução do Erro Quadrático por Época\n{titulo}")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático")
plt.grid(True)
plt.show()

#gráfico 3D da separação
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plotando as entradas e as saídas
ax.scatter(X[Yd == 0, 0], X[Yd == 0, 1], X[Yd == 0, 2], color='b', label='Saída 0', s=100)
ax.scatter(X[Yd == 1, 0], X[Yd == 1, 1], X[Yd == 1, 2], color='r', label='Saída 1', s=100)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title(f'Separação das Entradas - {titulo}')
ax.legend()


plt.show()
