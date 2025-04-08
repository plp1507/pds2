import numpy as np
from matplotlib import pyplot as plt

#GRR202127676
Fs = 1e3
x = np.sin(2*np.pi*27*np.arange(0.4*Fs)/(Fs))

npt = len(x)

w = np.zeros([npt, npt], dtype='complex')

for i in range(npt):
    w[i] = np.exp(-2j*np.pi*np.arange(npt)/npt)**i

w /= npt

fft_x = np.transpose(np.matmul(w, np.transpose(x)))

fig, ax = plt.subplots(2, 1)

ax[0].plot(np.linspace(0, 1000, 400), abs(fft_x))
ax[1].plot(np.linspace(0, 1000, 400), np.angle(fft_x))

ax[0].set_xlabel('k')
ax[1].set_xlabel('k')

ax[0].set_ylabel('Magnitude')
ax[1].set_ylabel('Fase (rad)')

ax[0].grid()
ax[1].grid()
plt.show()
