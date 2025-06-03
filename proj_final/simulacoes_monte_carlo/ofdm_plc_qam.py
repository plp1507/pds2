"""
Simulação de Montecarlo para sistemas QAM OFDM através de canal PLC

Autor: Pedro Lucca
Data: 23/10/2024
Versão: 3.2
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.io import loadmat
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
N = 1000000
# Número de blocos de símbolo
L = 4
# Comprimento do prefíxo cíclico
Ncp = 3
# Ordem da constelação
M = 16
#Limites da SNR (dB)
snr_min = 0
snr_max = 100

# SNR (dB)
snr_dB = np.arange(snr_min, snr_max, 3)
# SNR linear
snr = 10**(snr_dB/10)
# Potência do ruído
sigma_2 = 1/snr


#%%Curva teórica da SER
'''
snr_dB_teorico = np.linspace(snr_min, snr_max, 1000)
snrT = 10**(snr_dB_teorico/10)
ser_teorica = np.zeros(len(snr_dB_teorico))
for i in range(len(snr_dB_teorico)):
    argerfc = (3/(2*(M-1)))*(snrT[i])
    p = (1-np.sqrt(1/M))*erfc(np.sqrt(argerfc))
    ser_teorica[i] = 1-((1-p)**2)
'''


#%% Carregamento do arquivo referente ao canal PLC
canal = loadmat("NB_0_500k.mat")
canal = canal['h'][0]
canal = np.concatenate((canal, np.zeros((2*N+Ncp)*L - len(canal))))


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

#convolução com a resposta ao impulso do canal
y_tilde = np.convolve(xn.real,canal)[:(2*N + Ncp)*L]


#%% Simulação Montecarlo
ser = np.zeros(len(snr))

for k, Noise in enumerate(sigma_2):
    
    erro = 0
    
    while(erro == 0):
        #geração do ruído a partir da SNR
        v = np.sqrt(Noise)*np.random.randn(len(y_tilde))

        #checagem da SNR obtida
        Pv = np.sum(np.abs(v)**2)/len(v)
    
        #adição de ruído
        y = y_tilde + v
    
        y = np.fft.ifft(np.fft.fft(y)/np.fft.fft(canal))
    
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
        
    ser[k]  = erro/(N*L)
   
    print('SNR = {} dB'.format(snr_dB[k]))
    print('SNR simulada = {} dB'.format(10*np.log10(1/Pv)))
    
    print('SER obtida = {}'.format(ser[k]))
    print("")


#%% Figuras
#SNRdB x SER teórica
'''
plt.figure(figsize=(6,3))
plt.plot(snr_dB_teorico, ser_teorica, label ='curva teórica')
'''
# SNRdB x SER simulada
plt.plot(snr_dB, ser, 'o', label ='pontos simulados')

plt.ylabel('SER')
plt.xlabel('SNR (dB)')

plt.yscale('log')

plt.grid()
plt.show()

