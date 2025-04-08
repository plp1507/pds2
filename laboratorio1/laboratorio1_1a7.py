import numpy as np
from matplotlib import pyplot as plt

####primeira questão

#a
ampl = 3 #amplitude
dec  = (-1/12)+(np.pi/6)*1j #decaimento
n = np.arange(50) #vetor tempo com 50 instantes
x = ampl*np.exp(dec*n)

fig, ax = plt.subplots(2, 1)
ax[0].stem(x.real)
ax[1].stem(x.imag)

ax[0].set_xlabel('n')
ax[0].set_ylabel('Amplitude - real')

ax[1].set_xlabel('n')
ax[1].set_ylabel('Amplitude - imaginária')

ax[0].grid()
ax[1].grid()

plt.show()

#b
dec = -np.conjugate(dec)
x = ampl*np.exp(dec*n)

fig, ax = plt.subplots(2, 1)
ax[0].stem(x.real)
ax[1].stem(x.imag)

ax[0].set_xlabel('n')
ax[0].set_ylabel('Amplitude - real')

ax[1].set_xlabel('n')
ax[1].set_ylabel('Amplitude - imaginária')

ax[0].grid()
ax[1].grid()

plt.show()

####segunda questão
x = 0.5*np.exp(np.arange(30)*0.8)

#a

plt.stem(x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

#b
x = 0.5*np.exp(np.arange(30)*0.2)

plt.stem(x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

#c
x = x**-1

plt.stem(x)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

####terceira questão

x = 3*np.cos(2*np.pi*0.08*np.arange(50))

#a

fig, ax = plt.subplots(3, 1)
ax[0].plot(x)
ax[1].stem(x)
ax[2].stairs(x)

for i in range(3):
    ax[i].set_xlabel('n')
    ax[i].set_ylabel('Amplitude')
    ax[i].grid()

plt.show()

#b
x = 1.5*np.cos(2*np.pi*0.5*np.arange(80))

fig, ax = plt.subplots(3, 1)
ax[0].plot(x)
ax[1].stem(x)
ax[2].stairs(x)

for i in range(3):
    ax[i].set_xlabel('n')
    ax[i].set_ylabel('Amplitude')
    ax[i].grid()

plt.show()

####quarta questão
x = 2*np.arange(50)*0.9**np.arange(50)

#a
x_ruido = x + np.random.randn(50)

#b
x_filtrado = np.convolve(x_ruido, np.ones(3)/3, 'same')

plt.plot(x, ':', label = 'sinal original')
plt.plot(x_ruido, label = 'sinal ruidoso')
plt.plot(x_filtrado, label = 'sinal filtrado')

plt.legend()
plt.grid()
plt.show()

####quinta questão
x = np.zeros(60); x[10:50] = 1

#a
y = np.convolve(x, x)

#b
z = np.convolve(x, y)

###normalização
y = y/np.max(y)
z = z/np.max(z)

plt.plot(np.arange(60), x, label = 'x')
plt.plot(np.arange(-29, 90), y, label = 'y')
plt.plot(np.arange(-58, 120), z, label = 'z')

plt.xlabel('n')

plt.legend()
plt.grid()
plt.show()

####sexta questão

x = np.cos(2*np.pi*0.4*np.arange(32))

fig, ax = plt.subplots(2, 1)
ax[0].stem(abs(np.fft.fft(x)))
ax[1].stem(np.angle(np.fft.fft(x)))

ax[0].set_xlabel('k (frequência)')
ax[0].set_ylabel('Magnitude')

ax[1].set_xlabel('k (frequência)')
ax[1].set_ylabel('Fase (rad)')

ax[0].grid()
ax[1].grid()

plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].stem(abs(np.fft.fft(x, 8)))
ax[1].stem(np.angle(np.fft.fft(x, 8)))

ax[0].set_xlabel('k (frequência)')
ax[0].set_ylabel('Magnitude')

ax[1].set_xlabel('k (frequência)')
ax[1].set_ylabel('Fase (rad)')

ax[0].grid()
ax[1].grid()

plt.show()

####sétima questão

#
#nao faço ideia de como plotar bem esse cara -> muitos pontos
#


x = np.sin(2*np.pi*300*np.arange(16e3)/8e3) + np.sin(2*np.pi*3e3*np.arange(16e3)/8e3)

#a

fig, ax = plt.subplots(2, 1)
ax[0].plot(x)
ax[1].stem(abs(np.fft.fft(x)/16e3))

ax[0].set_xlabel('n')
ax[1].set_xlabel('k')

ax[0].grid()
ax[1].grid()

plt.show()

#b
h1 = np.asarray([1, 2.05, 2.05, 1])
h2 = np.asarray([1, -2.05, 2.05, -1])

y1 = np.convolve(x, h1)
y2 = np.convolve(x, h2)

fig, ax = plt.subplots(2, 1)
ax[0].plot(y1)
ax[1].stem(abs(np.fft.fft(y1)/16e3))

ax[0].set_xlabel('n')
ax[1].set_xlabel('k')

ax[0].grid()
ax[1].grid()

plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(y2)
ax[1].stem(abs(np.fft.fft(y2)/16e3))

ax[0].set_xlabel('n')
ax[1].set_xlabel('k')

ax[0].grid()
ax[1].grid()

plt.show()

