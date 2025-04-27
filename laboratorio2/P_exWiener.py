import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def wiener_filter(inpt, dim, noise):
    box_filter = (1/(dim**2))*np.ones([dim, dim])
    mean_im = convolve2d(inpt, box_filter, 'same')
    
    var_im = convolve2d(inpt**2, box_filter, 'same') - mean_im**2

    noise = np.mean(var_im[:])

    var_im = np.max(var_im - noise, axis = 0)
    mean_im += var_im/(var_im + noise) * (inpt - mean_im)
    return mean_im, noise

#abrir imagem
image = plt.imread('Lenna.bmp')
image = np.mean(image, axis = 2)  #conversão pra escala de cinza

#adição de ruído
var = 0.025

w = np.sqrt(var)*np.random.randn(np.shape(image)[0], np.shape(image)[1])
im_ruido = image + w

#1a) diferentes tamanhos de filtros
f_im8, erro8 = wiener_filter(im_ruido, 8, 0)
f_im2, erro2 = wiener_filter(im_ruido, 2, 0)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image, cmap = 'gray')
ax[0].set_title('Imagem original')
ax[1].imshow(im_ruido, cmap = 'gray')
ax[1].set_title(f'Imagem com ruído de var. {var}')
plt.show()


fig, ax = plt.subplots(1, 2)
ax[0].imshow(f_im2, cmap = 'gray')
ax[0].set_ylabel('Filtro 2x2')
ax[0].set_xlabel(f'Erro: {np.round(erro2, 3)}')
ax[1].imshow(f_im8, cmap = 'gray')
ax[1].set_ylabel('Filtro 8x8')
ax[1].set_xlabel(f'Erro: {np.round(erro8, 3)}')
plt.show()

#1b
var_ = np.arange(0.015, 0.085, 0.01)

fig, ax = plt.subplots(2,2)
fig.suptitle('Filtro 5x5')

for i in range(len(var_)):
    w_ = np.sqrt(var_[i])*np.random.randn(np.shape(image)[0], np.shape(image)[1])
    im_ruido_ = image + w_
    
    f_im, erro = wiener_filter(im_ruido_, 5, 0)
    
    if(i < 4):    
        ax[i//2, i%2].imshow(f_im, cmap = 'gray')
        ax[i//2, i%2].set_xlabel(f'erro: {np.round(erro, 3)}')
        ax[i//2, i%2].set_ylabel(f'var. {round(var_[i], 3)}')
    if(i>=4):
        ax[(i-4)//2, (i-4)%2].imshow(f_im, cmap = 'gray')
        ax[(i-4)//2, (i-4)%2].set_xlabel(f'erro: {np.round(erro, 3)}')
        ax[(i-4)//2, (i-4)%2].set_ylabel(f'var. {round(var_[i], 3)}')

    if(i==3):
        plt.show()
        fig, ax = plt.subplots(2,2)
        fig.suptitle('Filtro 5x5')
    
    if(i==7):
        plt.show()

