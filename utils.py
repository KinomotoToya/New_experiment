import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


def get_padding(input_ndim, whole_dim, mask_dim):
    size00 = (whole_dim[0]-mask_dim[0])//2
    size01 = whole_dim[0]-mask_dim[0]-size00
    size10 = (whole_dim[1]-mask_dim[1])//2
    size11 = whole_dim[1]-mask_dim[1]-size10
    if input_ndim == 2:
        paddings = tf.constant([[size00, size01], [size10, size11]])
    elif input_ndim == 3:
        paddings = tf.constant([[0, 0], [size00, size01], [size10, size11]])
    else:
        paddings = tf.constant(
            [[0, 0], [size00, size01], [size10, size11], [0, 0]])
    # input_padding = tf.pad(input, paddings, 'CONSTANT')
    return paddings


def get_incoherent(input_image):
    input_size = input_image.shape[0]
    inputs_list = []
    for i in range(input_size):
        for j in range(input_size):
            if input_image[i, j] > 0:
                empty = np.zeros((input_size, input_size))
                empty[i, j] = math.sqrt(input_image[i, j])
                inputs_list.append(empty)
    return np.array(inputs_list)


def dct_2d(feature_map, norm=None):  # can also be 'ortho'
    X1 = tf.signal.dct(feature_map, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 2, 1])
    X2 = tf.signal.dct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 2, 1])
    return X2_t


def idct_2d(feature_map, norm=None):  # can also be 'ortho'
    X1 = tf.signal.idct(feature_map, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 2, 1])
    X2 = tf.signal.idct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 2, 1])
    return X2_t


"""
def lowpass_fft(phase, size):
    phase_fft = np.fft.rfft2(phase)
    #phase_fft = np.fft.fftshift(phase_fft)
    print(phase_fft.shape)

    h, w = phase.shape
    a, b = size
    low_pass_phase_fft = np.zeros_like(phase_fft)
    low_pass_phase_fft[(h-a)//2:(h+a)//2, (w-b)//2:(w+b) //
                       2] = phase_fft[(h-a)//2:(h+a)//2, (w-b)//2:(w+b)//2]

    #low_pass_phase = np.fft.ifftshift(low_pass_phase_fft)
    low_pass_phase = np.fft.irfft2(low_pass_phase_fft)
    plt.subplot(121)
    plt.imshow(np.abs(phase_fft))
    plt.subplot(122)
    plt.imshow(np.abs(low_pass_phase_fft))
    plt.show()
    return low_pass_phase
"""
