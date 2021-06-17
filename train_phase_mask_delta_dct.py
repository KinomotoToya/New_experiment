#########  Use a delta point source to fit PSF ###########

import tensorflow as tf
import numpy as np
import math
import os
import time
import argparse
import matplotlib.pyplot as plt
from DDNN import *
from utils import *
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import scipy.io as sio

H = sio.loadmat('H.mat')['H']
H = H[tf.newaxis, :36, :36]

whole_dim = 3176
mask_dim = (1080, 1920)

wave_lambda = 532e-9
focal_length = 6e-2
pixel_size = 8e-6

inputs = tf.keras.Input(shape=(whole_dim, whole_dim),
                        dtype=tf.complex64, name="input")
x = Propagation("conv1", whole_dim, pixel_size,
                focal_length, wave_lambda)(inputs)
x = PhaseMaskDCT("phase1", whole_dim, mask_dim, (1080, 1920))(x)
x = Propagation("conv3", whole_dim, pixel_size,
                focal_length, wave_lambda)(x)
x = tf.square(tf.abs(x))

ddnn = tf.keras.Model(inputs=inputs, outputs=x)
ddnn.summary()
#############################################
target_image = np.zeros((1, whole_dim, whole_dim))
target_image[0, whole_dim//2-1:whole_dim//2+1,
             whole_dim//2-1:whole_dim//2+1] = 1


kernels = np.load('kernels.npy')[:, :, 0, :]
size = 100
kernels = tf.image.resize(kernels, [size, size])
psf_gt = np.zeros((1, size*5, size*5), dtype=np.float32)
psf_gt[0, size:size*2, size:size*2] = kernels[:, :, 0]
psf_gt[0, size:size*2, size*3:size*4] = kernels[:, :, 1]
psf_gt[0, size*3:size*4, size:size*2] = kernels[:, :, 2]
psf_gt[0, size*3:size*4, size*3:size*4] = kernels[:, :, 3]
paddings = get_padding(3, (whole_dim, whole_dim), (size*5, size*5))
target_label = tf.pad(psf_gt, paddings)
"""
psf_gt = np.zeros((1, 60, 60), dtype=np.float32)
psf_gt[0, 40:60, 40:60] = 1
paddings = get_padding(3, (whole_dim, whole_dim), (60, 60))
target_label = tf.pad(psf_gt, paddings)
"""
now = time.strftime('%m%d_%H%M', time.localtime(time.time()))
folder = "%s_phase_mask" % now
if not os.path.exists(folder):
    os.makedirs(folder)
    os.makedirs(folder+'/temp_results')

target_image, target_label = target_image*1000, target_label*1000

csv_logger = CSVLogger(folder+'/training.log')
save_model = ModelCheckpoint(
    filepath=folder+'/{epoch:04d}.hdf5', verbose=1, monitor='loss', save_best_only=True, save_weights_only=True)
test = MyTestPhase(folder, target_image, target_label, 2, slice(1100, 2100))

ddnn.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.1), loss='mse')
ddnn.fit(target_image, target_label, epochs=2000,
         callbacks=[csv_logger, test])
