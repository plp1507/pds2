# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:13:12 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

sysorder = 8

N = 2000
inp = np.random.randn(N)
n = np.random.randn(N)

b, a = butter(2, 0.25) #Butterworth discreto

y = lfilter(b, a, inp) #sistema com filtro discreto

h = np.array([
    0.0976,
    0.2873,
    0.3360,
    0.2210,
    0.0964
])

n = n * (np.std(y) / (10 * np.std(n)))
d = y + n

d = np.asarray(d).flatten()

totallength = len(d)

# LMS
N_conv = 60
w = np.zeros(sysorder)
y_est = np.zeros_like(d)
e = np.zeros_like(d)


# Algoritmo LMS
for n_idx in range(sysorder-1, N_conv):
    u = inp[n_idx-sysorder+1 : n_idx+1][::-1]
    y_est[n_idx] = np.dot(w, u)
    e[n_idx] = d[n_idx] - y_est[n_idx]

    mu = 0.32 if n_idx < 20 else 0.15
    w = w + mu * u * e[n_idx]

for n_idx in range(N_conv, totallength):
    u = inp[n_idx-sysorder+1 : n_idx+1][::-1]
    y_est[n_idx] = np.dot(w, u)
    e[n_idx] = d[n_idx] - y_est[n_idx]

plt.figure()
plt.plot(d, label='Saída real')
plt.plot(y_est, 'r', label='Saída estimada')
plt.title('Saída do Sistema')
plt.xlabel('Amostras')
plt.ylabel('Saídas real e estimada')
plt.legend()
plt.grid()

plt.figure()
plt.semilogy(np.abs(e))
plt.title('Curva de Erro')
plt.xlabel('Amostras')
plt.ylabel('Erro')
plt.grid()

plt.figure()
plt.plot(h, 'k+', label='Pesos atuais')
plt.plot(w, 'r*', label='Pesos estimados')
plt.title('Comparação entre pesos atuais e estimados')
plt.legend()
plt.axis([0, 6, 0.05, 0.35])
plt.grid()

plt.show()
