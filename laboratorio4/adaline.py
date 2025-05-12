import numpy as np
import matplotlib.pyplot as plt

def adaline_training(training_parameters, func_parameters):

    #### Características do treinamento
    eta = training_parameters[0]
    n_epocas = training_parameters[1]
    tolerancia = training_parameters[2]

    #### Dados de entrada
    n_amostras = func_parameters[0] # número de pontos
    a = func_parameters[1]          # coeficiente angular da reta
    b = func_parameters[2]          # termo independente
    sigma2 = func_parameters[3]     # variância de ruído

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

        if((dif < tolerancia) or (epoca > n_epocas)):
            break

        print(f'Época: {epoca}')
        epoca += 1

    return Yw, W, X_, vetor_erros 

''' 
    #### Características do treinamento
eta = training_parameters[0]
n_epocas = training_parameters[1]
tolerancia = training_parameters[2]

    #### Dados de entrada
n_amostras = func_parameters[0] # número de pontos
a = func_parameters[1]          # coeficiente angular da reta
b = func_parameters[2]          # termo independente
sigma2 = func_parameters[3]     # variância de ruído
'''

Yw, W, X_, erro = adaline_training([0.1, 10**2, 1], [10**3, 3, -1, 0.01])

#### Visualização dos resultados
fig, ax = plt.subplots(2, 1)

ax[0].plot(Yw)
ax[0].plot(W[0]*X_ + W[1])
ax[0].grid()

ax[1].plot(erro[0])
ax[1].plot(erro[1])
ax[1].grid()
plt.show()

