"""
Simulação de Montecarlo para sistemas QAM OFDM em canal AWGN

Autor: Pedro Lucca
Data: 03/06/2025
Versão: 1
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from math import erfc


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

def DMTmap(inpt):
    """
    Função para o mapeamento DMT

    Parâmetros
    ----------
    inpt: Mensagem a ser mapeada.
    
    Saída
    -----
    out: Mensagem após o mapeamento DMT.

    """
    dmt1 = np.vstack((inpt[-1].real, inpt[:N-1]))
    dmt2 = np.vstack((inpt[-1].imag, np.flipud(np.conjugate(inpt[:N-1]))))
    dmtmap = np.vstack((dmt1, dmt2))
    return dmtmap

def DMTdemap(inpt):
    out = np.vstack((inpt[1:N], inpt[0] + 1j*inpt[N]))
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
    for i in range(M):
        XA = np.column_stack((msgR.real, msgR.imag))
        XB = np.column_stack((constel.real, constel.imag))
        distcs = cdist(XA, XB, metric = 'euclidean')
    return constel[np.argmin(distcs)]


#%% Parâmetros do sistema
# Número de subportadoras
N = 128
# Número de blocos de símbolo
L = 10**3
# Comprimento do prefíxo cíclico
Ncp = 0
# Ordem da constelação
M = 16

# SNR (dB)
SNR_dB = 1
# SNR linear
sigma_2 = 10**(-SNR_dB/10)

#%%Cálculo teórico da SER

argerfc = (3/(2*(M-1)))*(SNR_dB)
p = (1-np.sqrt(1/M))*erfc(np.sqrt(argerfc))
SERt = 1-((1-p)**2)

#%% Geração do sinal OFDM
# Geração da consteção M-PAM
constel = constellation(M)

#escolha aleatória de pontos da constelação, mundança serie/paralelo e mapeamento DMT
X = np.random.choice(constel, N*L)
Xl = np.reshape(X, [N, L], order = 'F')
Xmap = DMTmap(Xl)

#aplicação da IDFT, adição do prefx. cíclico e conversão paralelo/serie
x = np.fft.ifft(Xmap, axis = 0, norm = 'ortho')
xcp = np.vstack((x[2*N-Ncp:], x))
xn = np.reshape(xcp,-1, order = 'F')

y_tilde = xn.real

#%% Passagem pelo canal e recepção

#convolução com canal
sigma = 1/2
h = np.linspace(0, 3)
p_h = (h/sigma**2)*np.exp(-(h**2)/2*sigma**2)


#geração do ruído a partir da SNR
v = np.sqrt(sigma_2)*np.random.randn(len(y_tilde))

#checagem da SNR obtida
Pv = np.sum(np.abs(v)**2)/len(v)
    
#adição de ruído
y = y_tilde + v
    
#conversão serie/paralelo, remoção do prefx. cíclico, aplicação da DFT e desmapeamento
xcpR = np.reshape(y, [2*N+Ncp, L], order = 'F')
xR = np.delete(xcpR, range(Ncp), axis = 0)
XlR = np.fft.fft(xR, axis = 0, norm = 'ortho')
Xdemap = DMTdemap(XlR)
    
#conversão paralelo/série
XR = np.reshape(Xdemap, -1, order = 'F')
#detecção dos pontos após a passagem pelo canal
XRd = np.zeros(len(XR), dtype = "complex")
for i in range(N*L):
    XRd[i] = detec(XR[i])
        
#contagem do erro e SER
erro = np.sum(XRd != X)
        
SERs  = erro/(N*L)
   
print(f'SNR teórica:  {SNR_dB} dB')
print(f'SNR simulada: {round(10*np.log10(1/Pv), 3)} dB')
    
print(f'SER obtida:  {round(SERs, 3)}')
print(f'SER teórica: {round(SERt, 3)}')
print("")


#%% Figuras
plt.rcParams.update({'font.size':15})



