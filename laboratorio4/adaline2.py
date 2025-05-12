import numpy as np
import matplotlib.pyplot as plt

#### Características da entrada

a = -1    # coeficiente angular
b = 2     # termo independente

num_amostras = 1000     # n. de amostras

X = np.linspace(-2, 2, num_amostras)  # variável x
X = np.reshape(X, [num_amostras, 1])

Y = a*X + b + np.random.randn(num_amostras, 1) # adição de ruído

def adaline_training(Y, X, eta, tolerancia, max_epocas):

    # Adição do bias na entrada X e pesos W
    X = np.hstack((X, np.ones([len(X), 1]))) 
    W = np.random.randn(2)

    erro = 0
    vetor_erros = []
    epoca = 1

    random_indx = np.arange(len(X))
    
    #### Loop de aprendizado
    while(True):
        erroq = 0
        np.random.shuffle(random_indx)

        for i in range(len(X)):
            yh = np.matmul(W, np.reshape(X[random_indx[i]][:], [2, 1]))
            erro = Y[random_indx[i]] - yh
            dW = eta*erro*X[random_indx[i]][:]
            W += dW
            erroq += erro**2
  
        if(epoca != 1):
            vetor_erros = np.vstack((vetor_erros, erroq/num_amostras))
        else:
            vetor_erros.append(erroq/num_amostras)

        diferenca = np.abs(vetor_erros[epoca - 1] - erroq)
        if(diferenca < tolerancia or epoca > max_epocas):
            break
    
        epoca += 1

    return W, vetor_erros

#### Parte 1 do exercício 1
# Coeficiente de aprendizado
eta = np.arange(0.01, 0.1, 0.02)

# Tolerância de um erro
tolerancia = 1

# Núm. máximo de épocas
max_epocas = 10**2

## plot das retas
plt.plot(X, Y, label = 'reta com ruído', lw = '0.8')

for o in range(len(eta)):
    W, vetor_erros = adaline_training(Y, X, eta[o], tolerancia, max_epocas)
    plt.plot(X, W[0]*X + W[1], label = f'eta = {eta[o]}', lw = '3')

plt.grid()
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()

## plot dos erros
for o in range(len(eta)):
    W, vetor_erros = adaline_training(Y, X, eta[o], tolerancia, max_epocas)
    plt.semilogy(np.arange(max_epocas), np.reshape(vetor_erros[1:], -1), label = f'eta = {eta[o]}')

plt.grid()
plt.legend()
plt.ylabel('Erro quadrático médio')
plt.xlabel('Épocas')
plt.show()

#### Parte 2 do exercício 1
eta = 0.02
tolerancia = 1
max_epocas = np.arange(100, 1000+200, 200)

plt.plot(X, Y, label = 'reta com ruído', lw = '0.8')

for o in range(len(max_epocas)):
    W, vetor_erros = adaline_training(Y, X, eta, tolerancia, max_epocas[o])
    plt.plot(X, W[0]*X + W[1], label = f'máx. épocas = {max_epocas[o]}', lw = '3')

plt.grid()
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()

## plot dos erros
for o in range(len(max_epocas)):
    W, vetor_erros = adaline_training(Y, X, eta, tolerancia, max_epocas[o])
    plt.semilogy(np.arange(max_epocas[o]), np.reshape(vetor_erros[1:], -1), label = f'máx. épocas = {max_epocas[o]}')

plt.grid()
plt.legend()
plt.ylabel('Erro quadrático médio')
plt.xlabel('Épocas')
plt.show()

