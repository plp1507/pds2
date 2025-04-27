import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

sysorder = 8
N = 2000

inp = np.random.randn(N)
n = np.random.randn(N)

b, a = signal.butter(2, 0.25)

y = signal.lfilter(b, a, inp)

h = np.asarray([0.0976, 0.2873, 0.336, 0.221, 0.0964])

n = n*np.std(y)/(10*np.std(n))
d = y + n

totalL = len(d)

M = 60

w = np.transpose(np.zeros(sysorder))
e = np.zeros(M)
for i in range(sysorder, M):
    u = inp[i:-1:i-sysorder+1]
    y[i] = np.matmul(np.transpose(w),u)
    e[i] = d[i] - y[i]


