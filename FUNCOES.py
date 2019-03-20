import os
import numpy as np


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


# DITHERING - apply threshold
def apply_threshold(pixel, colors):
    return np.round(colors*pixel/255) * (255/colors)


# FLOYD–STEINBERG DITHERING
def dither(oimg, colors):
    img = oimg.copy()
    y_lim, x_lim = img.shape[0], img.shape[1]
    for y in range(y_lim):
        for x in range(x_lim):
            oldpixel = img[y][x].copy().astype('uint16')
            newpixel = apply_threshold(oldpixel.copy(), colors)
            img[y][x] = newpixel.copy()
            quant_error = oldpixel - newpixel
            if x < x_lim - 1:
                img[y][x + 1] += np.round(quant_error*(7./16)).astype('uint8')
            if x > 0 and y < y_lim - 1:
                img[y + 1][x - 1] += np.round(quant_error*(3./16)).astype('uint8')
            if y < y_lim - 1:
                img[y + 1][x] += np.round(quant_error*(5./16)).astype('uint8')
            if x < x_lim - 1 and y < y_lim - 1:
                img[y + 1][x + 1] += np.round(quant_error*(1./16)).astype('uint8')
    return img.astype('uint8')


# SAVE IMAGE TO BIN FILE
def save_img_to_bin(imgdata, filename):
    with open(filename, 'wb') as file:
        np.array([imgdata.shape[0], imgdata.shape[1]]).astype('uint16').tofile(file)
        imgdata.tofile(file)


# READ IMAGE FROM BIN FILE
def read_img_from_bin(filename):
    with open(filename, 'rb') as file:
        h = np.fromfile(file, 'uint16', 1)
        w = np.fromfile(file, 'uint16', 1)
        img = np.reshape(np.fromfile(file, 'uint8'), (int(h), int(w), 3))

    return img
