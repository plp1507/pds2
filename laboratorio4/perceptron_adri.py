# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:42:34 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt

print("__________________________________________________________________")
print(" Configurações:\n")
RealizarAnd = int(input("   >>> Entre com (1) para AND Lógico e (0) para OR Lógico: "))
print("__________________________________________________________________\n\n")
input("   >>> PRESSIONE ENTER PARA PROCESSAR O TREINAMENTO...")

if RealizarAnd:
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Yd = np.array([0, 0, 0, 1])
else:
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Yd = np.array([0, 1, 1, 1])

N = X.shape[1]
NumeroPadroes = X.shape[0]

X_bias = -1 * np.ones((NumeroPadroes, 1))
X = np.hstack((X, X_bias))

W = np.random.randn(1, N)
Bias = np.random.randn(1)
W = np.append(W, Bias)

Eta = 100000
Tolerancia = 0.001
Vetor_Erros = []

X1Aux = np.linspace(-0.5, 1.5, 100)

while True:
    Erroq = 0
    Idx = np.random.permutation(NumeroPadroes)

    for i in range(NumeroPadroes):
        xi = X[Idx[i], :]
        y = 1 if np.dot(W, xi) >= 0 else 0
        erro = Yd[Idx[i]] - y
        delta_W = Eta * erro * xi
        W += delta_W
        Erroq += erro**2

    Vetor_Erros.append(Erroq)

    plt.clf()
    if RealizarAnd:
        plt.plot(X[0:3, 0], X[0:3, 1], 'bo', label='Resultado do AND = 0')
        plt.plot(X[3, 0], X[3, 1], 'r*', label='Resultado do AND = 1')
        plt.title('Evolução da Reta de Separação - AND Lógico')
    else:
        plt.plot(X[0, 0], X[0, 1], 'bo', label='Resultado do OR = 0')
        plt.plot(X[1:, 0], X[1:, 1], 'r*', label='Resultado do OR = 1')
        plt.title('Evolução da Reta de Separação - OR Lógico')

    # Reta de separação: W0*x + W1*y + Wb = 0 → y = -(W0/W1)*x + Wb/W1
    if W[1] != 0:
        reta = -(W[0] / W[1]) * X1Aux + W[2] / W[1]
        plt.plot(X1Aux, reta, 'k--', label='Reta de Separação')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.axis([-0.5, 1.5, -0.5, 2])
    plt.legend(loc='upper left')
    plt.pause(0.25)

    if Erroq < Tolerancia:
        break


plt.figure()
plt.plot(Vetor_Erros, '^-')
plt.xlabel('Época')
plt.ylabel('Erro Quadrático')
plt.title('Erro Quadrático Acumulado por Época de Treinamento')
plt.grid(True)
plt.show()

print("__________________________________________________________________")
print(" Resultados:\n")
print(f"   >>> O valor obtido para o peso 1 foi: {W[0]:.6f}")
print(f"   >>> O valor obtido para o peso 2 foi: {W[1]:.6f}")
print(f"   >>> O valor obtido para o bias foi  : {W[2]:.6f}")
print("__________________________________________________________________")
