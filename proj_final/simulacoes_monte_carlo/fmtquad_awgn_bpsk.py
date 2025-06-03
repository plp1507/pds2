"""
Simulação de Monte Carlo de sistema FMT - BPSK

Autor: Pedro Lucca Pereira
Data:  26/01/2025
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf

#configurações da SNR
SNRt = np.linspace(0, 10, 1000)
SNRs = SNRt[::len(SNRt)//7]
sigma2 = 10**(-SNRs/10)

#curva teórica
SERt = 0.5 - 0.5*erf(np.sqrt(10**(SNRt/10)))

#número de símbolos
n = 10**6
#comprimento do bloco multiportadora
l = 11
#número de portadoras
n_port = 5

#pulso protótipo
p = np.ones(l); p /= np.linalg.norm(p)

#geração dos símbolos aleatórios BPSK com 2 portadoras
mensagem = 2*np.round(np.random.rand(n*n_port)) - 1
seq= np.reshape(mensagem, [n_port, n])

sup = np.zeros([n_port, l*n])
x = np.zeros([n_port, l*n], dtype = 'complex')

for i in range(n_port):
    #upsampling
    sup[i][::l] = seq[i]
    #passagem pelo filtro
    x[i] = np.convolve(sup[i], p)[:l*n]

#modulação do sinal
xb = x
for i in range(n_port):
    x[i] = xb[i] * np.exp(2j*np.pi*(4+i)*np.arange(l*n)/l)

x = np.sum(x, axis=0)

#recepção do sinal
q = p[::-1]
z = np.zeros([n_port, l*n], dtype = 'complex')
z_d = np.zeros([n_port, n], dtype = 'complex')

SERs = np.zeros(len(SNRs))

for k, noise in enumerate(sigma2):

    w = np.sqrt(noise)*(np.random.randn(n*l))
    y = x + w
    
    for i in range(n_port):
        xb[i] = y*np.exp(-2j*np.pi*(4+i)*np.arange(n*l)/l)
        z[i] = np.convolve(xb[i], q)[:l*n]
        #remoção do atraso de grupo e downsampling
        z_d[i] = z[i][2*int((l-1)/2)::l]

    fin = np.reshape(np.sign(z_d.real), -1)
    SERs[k] = np.sum(fin != mensagem)/(n*n_port)
    print(f'SNR: {SNRs[k]}dB')
    print(f'SNR calculada: {10*np.log10(np.mean(abs(fin)**2)/np.mean(abs(w)**2))}dB')
    print(f'SER obtida: {SERs[k]}\n')

plt.semilogy(SNRt, SERt, label = 'curva teórica')
plt.semilogy(SNRs, SERs, 'o:', label = 'pontos simulados')

plt.xlabel('SNR (dB)')
plt.ylabel('SER')

plt.title(f'Taxa de erro de símbolo/bit de sistema BPSK com {n_port} portadoras')

plt.grid()
plt.legend()
plt.show()
