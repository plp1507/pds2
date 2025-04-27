# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:45:56 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

sysorder = 8
N = 2000
inp = np.random.randn(N)
n = np.random.randn(N)

b, a = butter(2, 0.25)

y = lfilter(b, a, inp)
h = np.array([0.0976, 0.2873, 0.3360, 0.2210, 0.0964])

n = n * (np.std(y) / (10 * np.std(n))) 
d = y + n  

totallength = len(d)
mu_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # Diferentes valores de mu
errors = {}

for mu in mu_values:
    w = np.zeros(sysorder) 
    y_est = np.zeros_like(d)
    e = np.zeros_like(d)  
    N_conv = 60  
    
    # Algoritmo LMS
    for n_idx in range(sysorder-1, N_conv):
        u = inp[n_idx-sysorder+1 : n_idx+1][::-1]
        y_est[n_idx] = np.dot(w, u)
        e[n_idx] = d[n_idx] - y_est[n_idx]
        w = w + mu * u * e[n_idx]

    for n_idx in range(N_conv, totallength):
        u = inp[n_idx-sysorder+1 : n_idx+1][::-1]
        y_est[n_idx] = np.dot(w, u)
        e[n_idx] = d[n_idx] - y_est[n_idx]

    errors[mu] = np.abs(e)  # Salva o erro absoluto para cada mu

# Plotando os erros para diferentes valores de mu
plt.figure(figsize=(10, 6))
for mu, error in errors.items():
    plt.semilogy(error, label=f"mu = {mu}")
plt.title("Erro absoluto em função de diferentes valores de mu")
plt.xlabel("Amostras")
plt.ylabel("Erro absoluto (escala logarítmica)")
plt.legend()
plt.grid(True)
plt.show()
