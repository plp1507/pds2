"""
Simulação de Montecarlo para sistemas QAM OFDM em canal AWGN

Autor: Pedro Lucca
Data: 03/06/2025
Versão: 1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from math import erfc

plt.rcParams.update({'font.size':15})

#%% Funções
def constellation(m):
    """
    Função para gerar a constelação como vetor de tamanho m-QAM

    Parâmetros
    ----------
    m: Ordem da constelação.

    Saídas
    -------
    out: símbolos de uma constelação m-QAM.
    
    """
    
    range_ = int(np.sqrt(m))
    out = np.zeros([range_,range_], dtype = "complex")
    for i in range(range_):
        for j in range(range_):
            out[i][j] = 1*i + 1j*j
    
    out = np.reshape(out, m)

    #p/ centrar a const. em 0
    out -= (np.sqrt(m)-1)*(0.5 +0.5j)

    #normalização pela energia média da constelação
    energTotal = np.sum(np.abs(out)**2)/m
    out /= np.sqrt(energTotal)
    
    return out

def detec(msgR):
    """
    Função de detecção de símbolos

    Parâmetros
    ----------
    msgR: Símbolo a ser detectado.

    Saída
    -------
    Ponto da constelação mais próximo.

    """
    XA = np.column_stack((msgR.real, msgR.imag))
    XB = np.column_stack((constel.real, constel.imag))
    distcs = cdist(XA, XB, metric = 'euclidean')
    return constel[np.argmin(distcs, axis = 1)]


#%% Parâmetros do sistema
# Número de subportadoras
N = 64
# Número de blocos de símbolo
L = 10**2
# Comprimento do prefíxo cíclico
Ncp = 16
# Ordem da constelação
M = 16

# SNR (dB)
SNR_dB = 20
# SNR linear
sigma_2 = 10**(-SNR_dB/10)

#%%Cálculo teórico da SER

argerfc = (3/(2*(M-1)))*(10**(SNR_dB/10))
p = (1-np.sqrt(1/M))*erfc(np.sqrt(argerfc))
SERt = 1-((1-p)**2)

#%% Geração do sinal OFDM
# Geração da consteção M-PAM
constel = constellation(M)

#escolha aleatória de pontos da constelação, mundança serie/paralelo e mapeamento DMT
X = np.random.choice(constel, N*L)
Xl = np.reshape(X, [N, L], order = 'F')

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xl, axis = 0, norm = 'ortho')
xcp = np.vstack((x[N-Ncp:], x))
xn = np.reshape(xcp,-1, order = 'F')
xn /= np.mean(np.abs(xn)**2)

#%% Passagem pelo canal e recepção

#canal = loadmat('./NB_0_500k.mat')['h'][0]

#y_tilde = np.convolve(xn, canal)[:(N+Ncp)*L]


#convolução com canal
n_taps = 6

# partes real e imaginária > distribuiçao normal
hri = np.sqrt(1/2)*(np.random.randn(n_taps*L) + 1j*np.random.randn(n_taps*L))

# módulo do canal
sigma = 1/2
h = np.linspace(0, 4, 1000)
p_h = (h/sigma**2)*np.exp(-(h**2)/(2*sigma**2))
p_h /= np.sum(p_h)


#plot da distribuição

#plt.plot(h, p_h)
#plt.grid()
#plt.show()


canal = np.random.choice(h, size = n_taps*L, p=p_h)*hri/np.mean(np.abs(hri)**2)
canal = np.reshape(canal, [L, n_taps])

# passagem pelo canal
y_tilde = np.zeros((N+Ncp)*L + n_taps - 1, dtype = 'complex')

for i in range(L):
    y_tilde[i*(N+Ncp):(i+1)*(N+Ncp) + n_taps - 1] = np.convolve(xn[i*(N+Ncp):(i+1)*(N+Ncp)], canal[i])

y_tilde = y_tilde[:(N+Ncp)*L]

#geração do ruído a partir da SNR
v = np.sqrt(sigma_2/2)*(np.random.randn((N+Ncp)*L) + 1j*np.random.randn((N+Ncp)*L))

#checagem da SNR obtida
Pv = np.mean(np.abs(v)**2)
    
#adição de ruído
y = y_tilde + v

#conversão serie/paralelo, remoção do prefx. cíclico, aplicação da DFT e desmapeamento
xcpR = np.reshape(y, [N+Ncp, L], order = 'F')
xR = np.delete(xcpR, range(Ncp), axis = 0)
XlR = np.fft.fft(xR, axis = 0, norm = 'ortho')

XlR = np.reshape(XlR, -1, order = 'F')

Xeq = np.zeros(N*L, dtype = 'complex')
XRd = np.zeros(N*L, dtype = 'complex')

###### equalizador lms
n_lms = 8
lms_eq = np.zeros([n_lms, 1], dtype = 'complex')
#lms_eq = np.transpose(np.array([np.random.randn(n_lms) + 1j*np.random.randn(n_lms)]))

eta = 0.000001

epoca = 0
max_epocas = 1500

SERs = 0
lim_SER = 0.4

erro = 0
lim_erro = Pv
erro_t = np.zeros(max_epocas)

while (erro > lim_erro  or epoca < max_epocas):
    for i in range(n_lms, N*L):
        Xeq[i] = np.matmul(XlR[i - n_lms:i], lms_eq)[0]

        erro = (detec(Xeq[i]) - Xeq[i])[0]

        d_lms = eta*np.conj(erro)*np.transpose(np.array([X[i-n_lms:i][::-1]]))

        lms_eq += d_lms
    
    
    #detecção dos pontos após a passagem pelo canal
    XRd = (detec(Xeq))[0]
    SERs = np.sum(XRd != X)/(N*L)

    erro_t[epoca] = abs(erro)
    epoca += 1
    print(f'erro:  {abs(erro)}')
    print(f'delta: {np.reshape(d_lms, -1)}')
    print(f'epoca: {epoca}')
    print(f'SER:   {SERs}\n\n')

print(f'filtro final: {lms_eq}')

plt.semilogy(erro_t[erro_t != 0])
plt.grid()
plt.show()

