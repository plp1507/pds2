import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import scipy.fftpack
import scipy.signal as signal
#import sounddevice as sd
####oitava questão

freqs_x = [697, 770, 852, 941]
freqs_y = [1209, 1336, 1477]

fs = 8000 #taxa de amostragem 8kHz
duracao = 0.1
t= np.linspace(0, duracao, int(fs * duracao), endpoint=False)

#dtmf = np.zeros([4, 3, 8e2])

x_freq = np.asarray([])
y_freq = np.asarray([])

for i in range(4):
    for j in range(3):
        sinal = np.sin(2 * np.pi * freqs_x[i] * t) + np.sin(2 * np.pi * freqs_y[j] * t)
        #dtmf[i][j] = np.sin(2*np.pi*x_freq[i]*np.arange(8e2)/8e3) + np.sin(2*np.pi*y_freq[j]*np.arange(8e2)/8e3)
        plt.subplot(2, 1, 1)
        plt.plot(t[:200], sinal[:200])  
        plt.title(f'Sinal DTMF: Tecla {(i*3) + j + 1}')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.grid()
    
        Y = np.abs(scipy.fftpack.fft(sinal))[:len(sinal)//2]
        f = np.fft.fftfreq(len(sinal), 1/fs)[:len(sinal)//2]

        plt.subplot(2, 1, 2)
        plt.plot(f, Y)
        plt.title('Espectro de Frequência')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.grid()

        plt.tight_layout()
        plt.show()
    
        # Reproduzir o som
#        sd.play(sinal, fs)
 #       sd.wait()  # Aguarda o som terminar antes de ir para o próximo

####nona questão

h1 = np.asarray([0.00253628299721552, 0.00255348555988337, 0, -0.00863465184119861, -0.0210222026271213,-0.0242112092355797, 0, 0.0605565098224690, 0.144591562515657, 0.219130722316847,0.248999000983656, 0.219130722316847, 0.144591562515657, 0.0605565098224690, 0, -0.0242112092355797, -0.0210222026271213, -0.00863465184119861, 0, 0.00255348555988337, 0.00253628299721552])

h2 = np.asarray([-0.00255268288602145, -0.00256999668237861, 0, 0.00869048446327945, 0.0211581345345303, 0.0243677616154823, 0, -0.0609480749706223, -0.145526507689415, -0.220547646013415, 0.751827168906504, -0.220547646013415, -0.145526507689415, -0.0609480749706223, 0, 0.0243677616154823, 0.0211581345345303, 0.00869048446327945, 0, -0.00256999668237861, -0.00255268288602145])

fs, audio = wavfile.read('./DTMF1.wav')
audio = audio/np.max(np.abs(audio))

#b
y1 = np.convolve(audio, h1, mode = 'same')
y2 = np.convolve(audio, h2, mode ='same')

def plot_signal(titulo, sinal, fs):
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(sinal)
    plt.title(f"{titulo} - Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid()

    Y = np.abs(np.fft.fft(sinal))
    freqs = np.fft.fftfreq(len(sinal), d=1/fs)

    plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs)//2], Y[:len(Y)//2])
    plt.title(f"{titulo} - Domínio da Frequência")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_signal("Filtro H1", h1, fs)
plot_signal("Filtro H2", h2, fs)

plot_signal("Sinal y1 (audio * H1)", y1, fs)
plot_signal("Sinal y2 (audio * H2)", y2, fs)

