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
from scipy.fft import fftn, ifftn
from scipy.signal import fftconvolve

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

def psf2otf(psf, shape):

    psf_pad = np.zeros(shape)
    insert_slices = tuple(slice(0, s) for s in psf.shape)
    psf_pad[insert_slices] = psf
    
    # Circularly shift psf to center
    for axis, axis_size in enumerate(psf.shape):
        psf_pad = np.roll(psf_pad, -int(axis_size // 2), axis=axis)
    
    otf = fftn(psf_pad)
    
    return otf

def _wiener_deconvolution(im, psf, nsr):
    PSF = psf2otf(psf, im.shape)
    PSF_abs_sq = np.abs(PSF) ** 2

    if np.isscalar(nsr):
        nsr_value = nsr
    else:
        if nsr.shape != im.shape:
            raise ValueError("NSR array must match the image shape.")
        nsr_value = nsr

    denominator = PSF_abs_sq + nsr_value
    denominator[denominator == 0] = np.finfo(float).eps

    wiener_filter = np.conj(PSF) / denominator
    deconvolved = ifftn(wiener_filter * fftn(im))
    return np.real(deconvolved)

def deconvwnr(img, psf, nsr=0):
    """
    Deconvolução de Wiener com verificações básicas.
    """
    if img.dtype.kind not in 'fcdui':
        raise ValueError("'img' deve ser um array numérico.")

    if not np.isrealobj(psf) or not np.issubdtype(psf.dtype, np.floating):
        raise ValueError("'psf' deve ser real e float.")

    if psf.ndim > img.ndim:
        raise ValueError("'psf' não pode ter mais dimensões que 'img'.")

    if any(p > i for p, i in zip(psf.shape, img.shape)):
        raise ValueError("Dimensões do PSF não podem exceder as da imagem.")

    if np.any(np.asarray(nsr) < 0):
        raise ValueError("'nsr' deve ser não-negativo.")

    img = img.astype(np.float64, copy=False)
    psf = psf.astype(np.float64, copy=False)
    if not np.isscalar(nsr):
        nsr = np.asarray(nsr, dtype=np.float64)


    return _wiener_deconvolution(img, psf, nsr)


# PARTE 1 - TRATAMENTO DE IMAGEM BORRADA E COM RUIDO

# Abertura da imagem
I=plt.imread('Lenna.bmp')
if I.ndim == 3:
    I = np.mean(I, axis=2)
#I = np.mean(I, axis=2)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(I, cmap='gray')
ax[0].set_xlabel('Imagem Original')

# Simular desfoque de movimento
LEN = 21
THETA = 11
PSF = motion_blur_psf(LEN, THETA)
blurred = convolve2d(I, PSF, 'same', boundary='wrap') #aplicar desfoque

ax[1].imshow(blurred, cmap='gray')
ax[1].set_xlabel('Imagem Borrada')


# Restaurar imagem borrada (sem ruído)
#wnr1,_ = wiener_filter(blurred, 2, 0) #aqui tem que ser a deconvolução
blurred = (blurred - np.min(blurred)) / (np.max(blurred) - np.min(blurred)) #normalização
wnr1 = deconvwnr(blurred, PSF, nsr=0)
ax[2].imshow(wnr1, cmap='gray')
ax[2].set_xlabel('Imagem Restaurada (sem ruído)')

plt.show()

# Adicionar ruído gaussiano
#noise_mean = 0
noise_var = 0.0001
blurred_noisy = random_noise(blurred, mode='gaussian', var=noise_var)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(blurred_noisy, cmap='gray')
ax[0].set_xlabel('Imagem Borrada com Ruído')

# Primeira tentativa de restauração (sem NSR)
wnr2 = deconvwnr(blurred_noisy, PSF, nsr=0)
ax[1].imshow(wnr2, cmap='gray')
ax[1].set_xlabel('Restauracao - NSR = 0')

# Segunda tentativa de restauração (com NSR estimado)
I = I / 255.0 #normalização
signal_var = np.var(I)
nsr_est = noise_var / signal_var 
wnr3 = deconvwnr(blurred_noisy, PSF, nsr=nsr_est)

ax[2].imshow(wnr3, cmap='gray')
ax[2].set_xlabel('Restauracao - NSR Estimado')
plt.show()

#PARTE 2 - TRATAMENTO DE IMAGEM BORRADA E COM RUIDO DE QUANTIZAÇÃO

# Abertura da imagem como uint8
# Se tiver três canais (RGB), converter para cinza
if I.ndim == 3:
    I_gray = np.mean(I, axis=2)
else:
    I_gray = I
    
I_uint8 =  (I_gray * 255).astype(np.uint8)
#print(f'Tipo da imagem original: {I_uint8.dtype}')  # Deve ser uint8

# Desfoque com quantização (uint8)
blurred_quantized = convolve2d(I_uint8, PSF, mode='same', boundary='wrap')
blurred_quantized = np.clip(blurred_quantized, 0, 255).astype(np.uint8)
#print(f'Tipo da imagem borrada: {blurred_quantized.dtype}')  # Também uint8

# Restaurar imagem borrada e quantizada - NSR = 0
wnr4 = deconvwnr(img_as_float(blurred_quantized), PSF, 0)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(wnr4, cmap='gray')
ax[0].set_xlabel('Restauracao - Quantizada, NSR = 0')

# Restaurar imagem com estimativa de NSR
uniform_quantization_var = (1 / 256) ** 2 / 12
signal_var = np.var(img_as_float(I_uint8))
wnr5 = deconvwnr(img_as_float(blurred_quantized), PSF, nsr=nsr_est)
ax[1].imshow(wnr5, cmap='gray')
ax[1].set_xlabel('Restauracao - Quantizada, NSR Estimado')

plt.show()


# PARTE 3 - ESTUDO DO IMPACTO DO ERRO NA VARIÂNCIA DO RUÍDO


# Adicionar ruído gaussiano
noise_var = 0.0001
blurred_noisy = random_noise(blurred, mode='gaussian', var=noise_var)

# Mostrar imagem borrada com ruído
plt.figure()
plt.imshow(blurred_noisy, cmap='gray')
plt.title('Imagem Borrada + Ruído Gaussiano')
plt.show()

# Calcular variância do sinal (imagem original)
signal_var = np.var(I_gray)

# NSR verdadeiro
nsr_true = noise_var / signal_var

# Variação do NSR estimado: de -50% a +50% em passos de 10%
percent_variations = np.arange(-0.5, 0.6, 0.1)  # de -50% até +50% (inclui o 0%)

plt.figure(figsize=(15,10))

for idx, perc in enumerate(percent_variations):
    nsr_modified = nsr_true * (1 + perc)  # Alterar NSR
    restored = deconvwnr(blurred_noisy, PSF, nsr=nsr_modified)

    plt.subplot(3, 4, idx+1)
    plt.imshow(restored, cmap='gray')
    plt.title(f'NSR {"+" if perc>=0 else ""}{int(perc*100)}%')
    plt.axis('off')

plt.tight_layout()
plt.suptitle('Impacto do Erro no NSR na Restauração', y=1.02, fontsize=16)
plt.show()
