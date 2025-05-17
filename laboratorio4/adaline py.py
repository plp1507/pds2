import numpy as np
import matplotlib.pyplot as plt

def adaline_training(training_parameters, func_parameters, visualizar=False):

    eta = training_parameters[0]
    n_epocas = training_parameters[1]
    tolerancia = training_parameters[2]


    n_amostras = func_parameters[0]  
    a = func_parameters[1]           
    b = func_parameters[2]           
    sigma2 = func_parameters[3]      

    X_ = np.linspace(-2, 2, n_amostras).reshape(-1, 1)
    ruido = np.sqrt(sigma2) * np.random.randn(n_amostras, 1)
    Yd = a * X_ + b + ruido  

    X_bias = np.ones((n_amostras, 1))
    X = np.hstack((X_ , X_bias)) 
    
    W = np.random.randn(1, 2)  

    vetor_erros = []

    
    for epoca in range(1, n_epocas + 1):
        erro_q = 0
        indices = np.random.permutation(n_amostras)

        for i in indices:
            xi = X[i].reshape(1, -1) 
            yd = Yd[i]                
            y = W @ xi.T             
            erro = yd - y            

            delta_W = eta * erro * xi 
            W += delta_W

            erro_q += erro ** 2

        erro_medio = erro_q / n_amostras
        vetor_erros.append(erro_medio.item())

        if visualizar:
            plt.clf()
            plt.plot(X_, Yd, 'r.', label='Amostras')
            reta = X @ W.T
            plt.plot(X_, reta, 'k-', label='Reta Obtida')
            plt.title(f'Época {epoca}')
            plt.grid(True)
            plt.axis([-2, 2, np.min(Yd)-1, np.max(Yd)+1])
            plt.legend(loc='upper left')
            plt.pause(0.1)

        if epoca > 1:
            dif = abs(vetor_erros[-1] - vetor_erros[-2])
            if dif < tolerancia:
                break

    return Yd, W, X_, vetor_erros

Yd, W, X_, erros = adaline_training([0.05, 1000, 1e-6], [1000, 3, 5, 0.4], visualizar=True)

plt.figure()
plt.plot(X_, Yd, 'r.', label='Amostras')
plt.plot(X_, W[0, 0]*X_ + W[0, 1], 'k-', label='Reta Obtida')
plt.title('Resultado Final')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(erros, '^-')
plt.title('Erro Quadrático Médio por Época')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.grid(True)
plt.show()

print("__________________________________________________________________")
print("Resultados:")
print(f"   >>> O valor obtido para o peso 1 foi: {W[0, 0]:.6f}")
print(f"   >>> O valor obtido para o bias foi: {W[0, 1]:.6f}")
print("__________________________________________________________________")

