import numpy as np
import matplotlib.pyplot as plt

#### Características do treinamento
eta = 0.1
n_epocas = 100
tolerancia = 1 


#### Dados de entrada
n_amostras = 10000  # número de pontos
a = 4     # coeficiente angular da reta
b = 1     # termo independente

sigma2 = 0.1 # variância de ruído

#### Sinal de entrada
X_ = np.linspace(-2, 2, n_amostras)
X = np.reshape(X_, np.shape(X_)+(1,))

noise = np.sqrt(sigma2)*np.random.randn(n_amostras)
noise = np.reshape(noise, np.shape(noise)+(1,))


Yw = a*X + b + noise  # reta com ruído

#### Inicializações
X_bias = np.ones([n_amostras, 1])      # entrada de bias

X = np.hstack((X, X_bias))

W = np.random.randn(2)   # pesos aleatórios

erro = 0
vetor_erros = np.zeros([1, 2])

epoca = 1

#### Loop de treinamento

while(True):
    erro_q = 0
    random_indx = np.arange(n_amostras)
    np.random.default_rng().shuffle(random_indx)
    for i in range(n_amostras):
        y = np.transpose(W*X[random_indx[i]][:])

        erro = np.transpose(Yw[random_indx[i]] - y)

        deltaw_ = eta*erro*X[random_indx[i]][:]
        W += deltaw_

        erro_q += erro**2

    
    if(epoca != 1):
        vetor_erros = np.vstack((vetor_erros, erro_q/n_amostras))
    else:
        vetor_erros = erro_q/n_amostras

    dif = np.linalg.norm(vetor_erros[epoca - 1] - erro_q)
    print(dif)

    if((dif < tolerancia) or (epoca > n_epocas)):
        break

    epoca += 1


#### Visualização dos resultados

plt.plot(Yw)
plt.plot(W[0]*X_ + W[1])
plt.grid()
plt.show()
