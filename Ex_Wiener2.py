# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 19:52:22 2025

@author: adrie
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import color, img_as_float
from skimage.restoration import wiener
from skimage.util import random_noise

# Função para criar PSF de movimento
def motion_blur_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    x = np.linspace(-center, center, length)
    y = np.tan(np.deg2rad(angle)) * x
    for i in range(length):
        xi = int(center + x[i])
        yi = int(center + y[i])
        if 0 <= xi < length and 0 <= yi < length:
            psf[yi, xi] = 1
    psf /= psf.sum()
    return psf

#Função filtro wiener

def wiener_filter(inpt, dim, noise):
    box_filter = (1/(dim**2))*np.ones([dim, dim])
    mean_im = convolve2d(inpt, box_filter, 'same')
    
    var_im = convolve2d(inpt**2, box_filter, 'same') - mean_im**2

    noise = np.mean(var_im[:])

    var_im = np.max(var_im - noise, axis = 0)
    mean_im += var_im/(var_im + noise) * (inpt - mean_im)
    return mean_im, noise

# PARTE 1 - TRATAMENTO DE IMAGEM BORRADA E COM RUIDO

# Abertura da imagem
I=plt.imread('Lenna.bmp')
I = np.mean(I, axis=2)

plt.figure()
plt.imshow(I, cmap='gray')
plt.title('Imagem Original')

# Simular desfoque de movimento
LEN = 21
THETA = 11
PSF = motion_blur_psf(LEN, THETA)
blurred = convolve2d(I, PSF, 'same', boundary='wrap')

plt.figure()
plt.imshow(blurred, cmap='gray')
plt.title('Imagem Borrada')


# Restaurar imagem borrada (sem ruído)
#wnr1,_ = wiener_filter(blurred, 2, 0) #aqui tem que ser a deconvolução
wnr1 = wiener(blurred, PSF, 0)
plt.figure()
plt.imshow(wnr1, cmap='gray') 
plt.title('Imagem Restaurada (sem ruído)')

plt.show()

# # Adicionar ruído gaussiano
noise_mean = 0
noise_var = 0.0001
blurred_noisy = random_noise(blurred, mode='gaussian', mean=noise_mean, var=noise_var)

# plt.figure()
# plt.imshow(blurred_noisy, cmap='gray')
# plt.title('Imagem Borrada com Ruído')
# plt.show()

# # Primeira tentativa de restauração (sem NSR)
# wnr2 = wiener(blurred_noisy, PSF, 0)
# plt.figure()
# plt.imshow(wnr2, cmap='gray')
# plt.title('Restauracao - NSR = 0')
# plt.show()

# # Segunda tentativa de restauração (com NSR estimado)
# signal_var = np.var(I)
# wnr3 = wiener(blurred_noisy, PSF, noise_var / signal_var)

# plt.figure()
# plt.imshow(wnr3, cmap='gray')
# plt.title('Restauracao - NSR Estimado')
# plt.show()

#PARTE 2 - TRATAMENTO DE IMAGEM BORRADA E COM RUIDO DE QUANTIZAÇÃO

# Abertura da imagem como uint8
# I_uint8 = cv2.imread('Lenna.bmp', cv2.IMREAD_GRAYSCALE)
# print(f'Tipo da imagem original: {I_uint8.dtype}')  # Deve ser uint8

# # Desfoque com quantização (uint8)
# blurred_quantized = cv2.filter2D(I_uint8, -1, PSF)
# print(f'Tipo da imagem borrada: {blurred_quantized.dtype}')  # Também uint8

# # Restaurar imagem borrada e quantizada - NSR = 0
# wnr4 = wiener(img_as_float(blurred_quantized), PSF, 0)
# plt.figure(); plt.imshow(wnr4, cmap='gray'); plt.title('Restauracao - Quantizada, NSR = 0')

# # Restaurar imagem com estimativa de NSR
# uniform_quantization_var = (1 / 256) ** 2 / 12
# signal_var = np.var(img_as_float(I_uint8))
# wnr5 = wiener(img_as_float(blurred_quantized), PSF, uniform_quantization_var / signal_var)
# plt.figure(); plt.imshow(wnr5, cmap='gray'); plt.title('Restauracao - Quantizada, NSR Estimado')

# plt.show()
