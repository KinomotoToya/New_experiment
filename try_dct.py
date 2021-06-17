import matplotlib.pyplot as plt
import numpy as np
from utils import dct_2d, idct_2d
import tensorflow as tf


def lowpass_dct(phase, size):
    lh, lw = size
    phase_dct = dct_2d(phase)
    lowpass_phase_dct = np.zeros_like(phase_dct)
    lowpass_phase_dct[0, :lh, :lw] = phase_dct[0, :lh, :lw]
    lowpass_phase = idct_2d(lowpass_phase_dct)
    return phase_dct, lowpass_phase_dct, lowpass_phase


phase = np.load('0616_1656_phase_mask/param_2000.npy')
phase = phase[tf.newaxis, :, :]
phase_dct, lowpass_phase_dct, lowpass_phase = lowpass_dct(
    phase, size=(300, 600))

plt.subplot(221)
plt.imshow(phase_dct[0])
plt.subplot(222)
plt.imshow(lowpass_phase_dct[0])
plt.subplot(223)
plt.imshow(phase[0])
plt.subplot(224)
plt.imshow(lowpass_phase[0])
plt.show()
