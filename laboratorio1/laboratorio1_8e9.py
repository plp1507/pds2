import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
####nona questão

h1 = np.asarray([0.00253628299721552, 0.00255348555988337, 0, -0.00863465184119861, -0.0210222026271213,-0.0242112092355797, 0, 0.0605565098224690, 0.144591562515657, 0.219130722316847,0.248999000983656, 0.219130722316847, 0.144591562515657, 0.0605565098224690, 0, -0.0242112092355797, -0.0210222026271213, -0.00863465184119861, 0, 0.00255348555988337, 0.00253628299721552])

h2 = np.asarray([-0.00255268288602145, -0.00256999668237861, 0, 0.00869048446327945, 0.0211581345345303, 0.0243677616154823, 0, -0.0609480749706223, -0.145526507689415, -0.220547646013415, 0.751827168906504, -0.220547646013415, -0.145526507689415, -0.0609480749706223, 0, 0.0243677616154823, 0.0211581345345303, 0.00869048446327945, 0, -0.00256999668237861, -0.00255268288602145])

fs, audio = wavfile.read('./DTMF1.wav')

#a
##ja faço
t, f, spectr = spectrogram(audio, fs)

plt.pcolormesh(f, t, spectr, shading='gouraud')
plt.xlabel('Tempo (s)')
plt.ylabel('Frequência (Hz)')
plt.show()

#b
y1 = np.convolve(audio, h1)
y2 = np.convolve(audio, h2)

#
#falta plotar essas coisa ai
#
