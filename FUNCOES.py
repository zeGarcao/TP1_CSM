import os
import numpy as np
import cv2


# CALCULAR TAXA DE COMPRESSAO
def tcompress(oimg_path,cimg_path):
    return os.path.getsize(oimg_path)/os.path.getsize(cimg_path)


# CALCULAR SNR
def SNR(oimg, ruido):
    Psinal = np.sum(np.power(oimg, 2.))
    Pruido = np.sum(np.power(ruido, 2.))
    return 10.*np.log10(Psinal/Pruido)


# CALCULAR PSNR
def PSNR(oimg, yimg, maxpval):
    MSE = np.sum(np.abs(oimg - yimg)**2.)/(len(oimg)*len(oimg[0]))
    MAXi = maxpval
    return 10.*np.log10((MAXi**2.)/MSE)


# OBTER IMAGEM A PARTIR DE UM SO BIT POR PIXEL
def onebitimg(img, bit):
    return ((img & np.power(2, bit))/np.power(2, bit)).astype('uint8')


# DITHERING
def dither(img):
    for y in range(0, len(img)):
        for x in range(0, len(img[0])):
            oldpixel = img[y][x]
            newpixel = np.round(oldpixel/255.)
            img[y][x] = newpixel
            quant_error = oldpixel - newpixel

            img[y][x+1] = img[y][x+1] + quant_error*(7./16)
            img[y+1][x-1] = img[y+1][x-1] + quant_error*(3./16)
            img[y+1][x] = img[y+1][x] + quant_error*(5./16)
            img[y+1][x+1] = img[y+1][x+1] + quant_error*(1./16)
