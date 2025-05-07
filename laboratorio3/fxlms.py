import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
'''
#### Duração da simulação
T = 10**4

#### Sistemas P(z) e S(z)
P = np.asarray([0.0723, 0.578, 0.367, 0.2210, 0.0883])
S = np.asarray([0.0875, 0.443, 0.254, 0.3210, 0.0964])

#### QUESTÃO 1 - Estimação de S (Ŝ ou Shw)
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

#### QUESTÃO 2 - Utilização de Ŝ(z) (Shw(z))
len_inpt = 1000

x_inpt = np.sqrt(0.002)*np.random.randn(len_inpt) #ruído branco gaussiano de variância 0.002
y = lfilter(S, 1, x_inpt)
yh= lfilter(Shw, 1, x_inpt)

plt.plot(y, label = 'saída com filtro real')
plt.plot(yh, label = 'saída com filtro estimado')
plt.grid()
plt.legend()
plt.ylabel('Amplitude')
plt.xlabel('k')
plt.show()
'''

#### QUESTÃO 3 - Utilização do filtro em sinal real
_, sig = wavfile.read('Flauta.wav')  # sinal de interesse
sig = sig/np.max(np.abs(sig))

w = np.sqrt(0.01)*np.random.randn(len(sig))     # vetor de ruído
noise_p = np.sum(w**2)

sig_inpt = sig + w   # sinal com ruído (entrada do sistema)
'''
plt.plot(sig_inpt, label = 'sinal ruidoso')
plt.plot(sig, label = 'sinal original')
plt.ylabel('Amplitude')
plt.xlabel('k')
plt.grid()
plt.legend()
plt.show()
'''
#algoritmo LMS
f_order = 10

Shx = np.zeros(f_order)
Shw = np.zeros(f_order)

e_canc = np.zeros(len(sig))

mu_max = 2/noise_p
print(f"mu máximo: {mu_max}")
mu_max = 0.025/noise_p
print(f'mu utilizado: {mu_max}')

plt.stem(np.arange(0 + 1, f_order + 1, 1), Shw, 'C2', label = 'pesos iniciais')

for i in range(f_order, len(sig)):
    for j in range(f_order):
        Shx[j] = sig_inpt[i-j]
        Shy = np.sum(Shw * Shx)
        e_canc[i] = sig[i] - Shy
        Shw[j] += mu_max*e_canc[i]*Shx[j]
    if(i % 8000 == 0):
        plt.stem(np.arange(i/(2.1*len(sig)) + 1, f_order + i/(2.1*len(sig)) + 1, 1), Shw, markerfmt = 'x')

step = 0.5
plt.stem(np.arange(step + 1, f_order + step + 1, 1), Shw, 'C1', label = 'pesos finais')
plt.grid()
plt.legend()
plt.show()

yh = lfilter(Shw, 1, sig_inpt)

fig, ax = plt.subplots(2, 1)
ax[0].plot(sig_inpt, label = 'sinal ruidoso')
ax[0].plot(yh, label = 'sinal estimado')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('k')
ax[0].grid()
ax[0].legend()
ax[1].stem(Shw, label = 'pesos do filtro')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Numerando dos coeficientes do filtro')
ax[1].grid()
ax[1].legend()

plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(yh, label = 'sinal estimado')
ax[0].plot(sig, label = 'sinal original')
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('k')
ax[0].grid()
ax[0].legend()

ax[1].semilogy(10*e_canc, label = 'erro de cancelamento final')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('k')
ax[1].grid()
ax[1].legend()

plt.show()

