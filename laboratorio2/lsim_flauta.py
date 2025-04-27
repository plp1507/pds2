# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:47:54 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def add_gaussian_noise(signal, noise_variance):
    noise = np.random.normal(0, np.sqrt(noise_variance), signal.shape)
    return signal + noise

fs, signal = wavfile.read('flauta.wav')  

signal = signal / np.max(np.abs(signal), axis=0)
noise_variance = 0.0250
noisy_signal = add_gaussian_noise(signal, noise_variance)

sysorder = 8
N = len(noisy_signal)  

w = np.zeros(sysorder)  # Pesos do filtro
y_est = np.zeros_like(noisy_signal)  # Saídas estimadas
e = np.zeros_like(noisy_signal)  # Erro
N_conv = 60  # Número de pontos para a convergência inicial

# Variando a taxa de convergência mu
mu_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
errors = {}

# Algoritmo LMS
for mu in mu_values:
    w = np.zeros(sysorder)  
    for n_idx in range(sysorder-1, N_conv):
        u = noisy_signal[n_idx-sysorder+1 : n_idx+1][::-1]
        y_est[n_idx] = np.dot(w, u)  
        e[n_idx] = noisy_signal[n_idx] - y_est[n_idx]  
        w = w + mu * u * e[n_idx]  

    for n_idx in range(N_conv, N):
        u = noisy_signal[n_idx-sysorder+1 : n_idx+1][::-1]
        y_est[n_idx] = np.dot(w, u) 
        e[n_idx] = noisy_signal[n_idx] - y_est[n_idx]  

    errors[mu] = np.abs(e)  


plt.figure(figsize=(12, 6))
plt.plot(noisy_signal[:1000], label="Sinal com Ruído", alpha=0.7)
plt.plot(signal[:1000], label="Sinal Original", alpha=0.7)
plt.plot(y_est[:1000], label="Sinal Estimado (Filtrado)", alpha=0.7)
plt.title("Comparação entre Sinal Original, Sinal com Ruído e Sinal Estimado")
plt.xlabel("Amostras")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for mu, error in errors.items():
    plt.semilogy(error, label=f"mu = {mu}")
plt.title("Erro absoluto em função de diferentes valores de mu")
plt.xlabel("Amostras")
plt.ylabel("Erro absoluto (escala logarítmica)")
plt.legend()
plt.grid(True)
plt.show()

filtered_signal = np.int16(y_est * 32767)  # Convertendo para formato de áudio 16 bits
wavfile.write('flauta_filtrada.wav', fs, filtered_signal)


