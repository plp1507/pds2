import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import erfc
from scipy.special import hyp2f1, gamma

# Parâmetro m da distribuição Nakagami
m = [.5, 1, 2, 15]

#### Características da SNR
npt = 12       # número de pontos da simulação
SNRt = np.linspace(-2, 10, 1000)
SNRs = SNRt[::len(SNRt)//(npt-1)]

SNRtlin = 10**(SNRt/10)     # valores lineares
sigma2 = 10**(-SNRs/10)

SERt = np.zeros([4, len(SNRt)])
#### Curvas teóricas Nakagami-m
for o in range(4):
    SERt[o] = (gamma(m[o]+.5)/(2*np.sqrt(np.pi)*gamma(m[o]+1)))*(((1 + SNRtlin/m[o]))**(-m[o]))*(hyp2f1(m[o], .5, m[o]+1, 1/(1+SNRtlin/m[o])))

SERs = np.zeros(npt)

#### Geração dos símbolos BPSK da mensagem
n = 10**5     # no. de símbolos
mensagem = 2*np.round(np.random.rand(n)) - 1

fig, ax = plt.subplots(2, 1)

ax[0].semilogy(SNRt, (1/2)*erfc(np.sqrt(SNRtlin)), '--', label = 'AWGN') #SER AWGN

#### Simulação de Monte Carlo
for o in range(4):
    channel_ = (2*(m[o]**m[o])*np.linspace(0, 3, n)**(2*m[o] - 1))*np.exp(-m[o]*np.linspace(0, 3, n)**2)/gamma(m[o]) # distribuição Nakagami-m
    channel = np.asarray(random.choices(np.linspace(0.000000001, 3, n), channel_, k = n)) # coleta de amostras da distribuição

    ax[1].hist(channel, np.linspace(0, 3, n),label = f'm = {m[o]}')

    for k, noise in enumerate(sigma2):
        w = np.sqrt(noise/2)*np.random.randn(n)    # vetor de ruído gaussiano
    
        sinal_rx = channel*mensagem + w            # passagem pelo canal
        sinal_rx_demod = np.sign(sinal_rx/channel) # equalização

        SERs[k] = np.sum(sinal_rx_demod != mensagem)/n # contabilização dos erros
    
    ax[0].semilogy(SNRs, SERs, 'x:', label = f'$m_s$ = {m[o]}')
    ax[0].semilogy(SNRt, SERt[o], label = f'$m_t$ = {m[o]}')

ax[1].set_xlabel('Mag.')
ax[1].set_ylabel('No. de símbolos')
ax[1].legend()

ax[0].set_xlabel('SNR (dB)')
ax[0].set_ylabel('BER')

ax[1].grid()
ax[0].grid()
ax[0].legend()
plt.show()
