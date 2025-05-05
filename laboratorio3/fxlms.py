import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import lfilter

#### Duração da simulação
T = 10**4

#### Sistemas P(z) e S(z)
P = np.asarray([0.0723, 0.578, 0.367, 0.2210, 0.0883])
S = np.asarray([0.0875, 0.443, 0.254, 0.3210, 0.0964])

#### QUESTÃO 1 - Estimação de S (Ŝ ou Sh)
x_iden = np.random.randn(T)
y_iden = lfilter(S, 1, x_iden) ## Aplicação de S(z) como filtro FIR num vetor de ruído gaussiano

noise_p = np.sum(x_iden**2)  ## Potência de ruído

f_order = 16
Shx = np.zeros(f_order)
Shw = np.zeros(f_order)

e_iden = np.zeros(T) ## buffer de erro de identificação

mu_max = 2/noise_p  ## determinação do mu a partir da pot. de ruído de entrada
print(mu_max)
for i in range(T):
    for j in range(f_order):
        Shx[j] = x_iden[i-j]
        Shy = np.sum(Shx * Shw)
        e_iden[i] = y_iden[i] - Shy
        Shw[j] = Shw[j] + mu_max*e_iden[i]*x_iden[i-j]

figure, ax = plt.subplots(2, 1)
ax[0].semilogy(10*e_iden, label = 'erro de identificação')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('k - tempo discreto')
ax[0].grid()
ax[0].legend()
ax[1].stem(S, label = 'Coeficientes de S(z)')
ax[1].stem(Shw, 'C1', label = 'Coeficientes de Ŝ(z)')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Numerando dos coeficientes do filtro')
ax[1].legend()
ax[1].grid()

plt.show()
