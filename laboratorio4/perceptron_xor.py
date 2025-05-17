# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:53:26 2025

@author: adrie
# Treinamento da porta lógica XOR com Perceptron simples
"""

import numpy as np
import matplotlib.pyplot as plt

# Entradas (com 2 bits) e saídas desejadas (XOR)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Yd = np.array([0, 1, 1, 0])

NumeroPadroes = X.shape[0]
X_bias = -1 * np.ones((NumeroPadroes, 1))
X = np.hstack((X, X_bias))

W = np.random.randn(1, X.shape[1])

Eta = 0.1               # taxa de aprendizado
Tolerancia = 0.001      # critério de parada
Vetor_Erros = []        # para armazenar o erro por época

while True:
    ErroEpoca = 0
    indices = np.random.permutation(NumeroPadroes)

    for i in indices:
        xi = X[i]
        yd = Yd[i]

        y = int(np.dot(W, xi.T) >= 0)

        erro = yd - y
        W += Eta * erro * xi

        ErroEpoca += erro ** 2

    Vetor_Erros.append(ErroEpoca)

    if ErroEpoca < Tolerancia:
        break

print("\nPesos finais:")
print(f"  Peso 1: {W[0,0]:.4f}")
print(f"  Peso 2: {W[0,1]:.4f}")
print(f"  Bias  : {W[0,2]:.4f}")

plt.plot(Vetor_Erros, marker='o')
plt.title("Erro Quadrático por Época - Porta XOR")
plt.xlabel("Época")
plt.ylabel("Erro Quadrático")
plt.grid(True)
plt.show()
