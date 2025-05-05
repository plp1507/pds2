# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 18:00:53 2025

@author: adrie
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Parâmetros fornecidos
S = np.array([0.0875, 0.443, 0.254, 0.3210, 0.0964])
P = np.array([0.0723, 0.578, 0.367, 0.2210, 0.0883])

# LMS para identificação de S(z)
def lms_identification(S_true, mu, n_iter, noise_power):
    N = len(S_true)
    w = np.zeros(N)
    error = []
    
    for _ in range(n_iter):
        x = np.random.randn()  # ruído branco
        d = np.dot(S_true, np.roll(w, 1)) + np.sqrt(noise_power) * np.random.randn()
        y = np.dot(w, np.roll(w, 1))
        e = d - y
        w = w + 2 * mu * e * np.roll(w, 1)
        error.append(e**2)

    return w, error

# FxLMS para cancelamento
def fx_lms(x, d, S_hat, P, mu, n_iter):
    N = len(S_hat)
    W = np.zeros(N)
    S_hat_buffer = np.zeros(N)
    e = np.zeros(n_iter)
    
    for n in range(n_iter):
        # Propagação do sinal de entrada
        x_P = np.convolve(x[max(0, n-N+1):n+1], P, mode='full')[:N]
        x_S = np.convolve(x[max(0, n-N+1):n+1], S_hat, mode='full')[:N]
        
        y = np.dot(W, x_S[::-1])
        e[n] = d[n] - y
        
        # Atualiza W usando LMS
        W = W + 2 * mu * e[n] * x_S[::-1]
    
    return e

# Parâmetros de identificação
mu_ident = 0.01
n_iter_ident = 5000
noise_power = 0.001

# Identificação de S(z)
S_hat, error_ident = lms_identification(S, mu_ident, n_iter_ident, noise_power)

# Plots da identificação
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(10*np.log10(error_ident))
plt.title('Evolução do erro de identificação (dB)')
plt.xlabel('Iterações')
plt.ylabel('Erro (dB)')

plt.subplot(1,2,2)
plt.stem(S, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(S_hat, linefmt='r--', markerfmt='rx', basefmt=' ')
plt.legend(['S(z)', 'S_hat(z)'])
plt.title('Comparação dos Coeficientes')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.tight_layout()
plt.show()

# Segunda parte - Implementação do FxLMS
n_iter_fxlms = 10000
mu_fxlms = 0.0005

# Entrada: ruído branco
x = np.random.normal(0, 1, n_iter_fxlms)
# d(n): ruído passando por P(z)
d = np.convolve(x, P, mode='full')[:n_iter_fxlms] + np.sqrt(0.002) * np.random.randn(n_iter_fxlms)

# Aplicação do FxLMS
e = fx_lms(x, d, S_hat, P, mu_fxlms, n_iter_fxlms)

# Plot do erro
plt.figure()
plt.plot(10*np.log10(e**2))
plt.title('Evolução do erro de cancelamento (dB)')
plt.xlabel('Iterações')
plt.ylabel('Erro (dB)')
plt.show()

# Terceira parte - Aplicação ao áudio
fs, audio = wavfile.read('Flauta.wav')
audio = audio / np.max(np.abs(audio))  # Normaliza

# Adiciona ruído
noise = np.random.normal(0, np.sqrt(0.002), len(audio))
audio_noisy = audio + noise

# d(n) para áudio
d_audio = np.convolve(audio_noisy, P, mode='full')[:len(audio)]

# Aplica FxLMS ao áudio
e_audio = fx_lms(audio_noisy, d_audio, S_hat, P, mu_fxlms, len(audio))

# Resultado
plt.figure()
plt.plot(audio, label='Original')
plt.plot(e_audio, label='Áudio após cancelamento')
plt.legend()
plt.title('Comparação do áudio original e do áudio após cancelamento')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.show()

# Salva áudio cancelado
e_audio_norm = np.int16(e_audio / np.max(np.abs(e_audio)) * 32767)
wavfile.write('audio_cancelado.wav', fs, e_audio_norm)
