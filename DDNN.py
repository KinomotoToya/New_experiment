import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.initializers import Constant
from utils import get_padding, idct_2d


class Lens(tf.keras.layers.Layer):
    def __init__(self, name, whole_dim, pixel_size, focal_length, wave_lambda, diameter=25.4e-3):
        super(Lens, self).__init__(name=name)
        # basic parameters
        x = np.arange((-np.ceil((whole_dim - 1) / 2)),
                      np.floor((whole_dim - 1) / 2)+0.5)
        xx, yy = np.meshgrid(x, x)
        temp1 = math.pi * pixel_size * pixel_size / wave_lambda / focal_length
        self.lens_function = np.exp(-1j * temp1 * (xx ** 2 + yy ** 2))

        if diameter > 0:
            num = int(diameter/pixel_size)
            pupil = (xx**2 + yy**2 <= (num//2)**2)
            self.lens_function[pupil == 0] = 0

    def call(self, input):
        out = tf.multiply(input, self.lens_function)
        return out


class Propagation(tf.keras.layers.Layer):
    def __init__(self, name, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Propagation, self).__init__(name=name)
        # init frequency parameter
        F = np.arange((-np.ceil((whole_dim - 1) / 2)),
                      np.floor((whole_dim - 1) / 2)+0.5) / pixel_size / whole_dim
        Fhh, Fvv = np.meshgrid(F, F)
        temp1 = 2.0 * math.pi * focal_length
        temp2 = np.complex64(1.0 / wave_lambda ** 2 -
                             Fvv ** 2.0 - Fhh ** 2.0) ** 0.5
        self.H = np.exp(1j * temp1 * temp2)

    def call(self, input):
        # transform input image to spectrum
        spectrum = tf.signal.fft2d(input)
        spectrum = tf.signal.fftshift(spectrum)
        # apply transfer function
        spectrum_z = tf.math.multiply(spectrum, self.H)
        # apply ifft
        spectrum_z = tf.signal.ifftshift(spectrum_z)
        img_z = tf.signal.ifft2d(spectrum_z)
        return img_z


class PhaseMask(tf.keras.layers.Layer):  # 经试验，self.A有没有都能收敛
    def __init__(self, name, whole_dim, mask_dim, phase=None):
        super(PhaseMask, self).__init__(name=name)
        if type(whole_dim) == int:
            whole_dim = (whole_dim, whole_dim)
        if type(mask_dim) == int:
            mask_dim = (mask_dim, mask_dim)

        self.w = self.add_weight("w",
                                 shape=(1,)+mask_dim,
                                 initializer=Constant(
                                     phase) if phase is not None else "random_normal",
                                 trainable=True,
                                 dtype=tf.float32
                                 )
        self.A = self.add_weight("A", shape=(1,),
                                 initializer="random_normal",
                                 trainable=True,
                                 dtype=tf.float32)
        self.paddings = get_padding(3, whole_dim, mask_dim)

    def call(self, input):
        mask_phase = tf.sigmoid(self.w) * 1.999 * math.pi
        mask_complex = tf.complex(
            self.A*tf.cos(mask_phase), self.A*tf.sin(mask_phase))
        mask = tf.pad(mask_complex, self.paddings, 'CONSTANT')
        out = tf.multiply(input, mask)
        return out


class PhaseMaskDCT(tf.keras.layers.Layer):
    def __init__(self, name, whole_dim, mask_dim, dct_dim, dct_value=None):
        super(PhaseMaskDCT, self).__init__(name=name)
        if type(whole_dim) == int:
            whole_dim = (whole_dim, whole_dim)
        if type(mask_dim) == int:
            mask_dim = (mask_dim, mask_dim)
        if type(dct_dim) == int:
            dct_dim = (dct_dim, dct_dim)

        self.w = self.add_weight("w",
                                 shape=(1,)+dct_dim,
                                 initializer=Constant(
                                     dct_value) if dct_value is not None else "random_normal",
                                 trainable=True,
                                 dtype=tf.float32
                                 )
        self.A = self.add_weight("A", shape=(1,),
                                 initializer="random_normal",
                                 trainable=True,
                                 dtype=tf.float32)

        self.paddings1 = tf.constant(
            [[0, 0], [0, mask_dim[0]-dct_dim[0]], [0, mask_dim[1]-dct_dim[1]]])
        self.paddings2 = get_padding(3, whole_dim, mask_dim)

    def call(self, input):
        mask_dct = tf.pad(self.w, self.paddings1, 'CONSTANT')
        mask_phase = idct_2d(mask_dct)
        mask_phase = tf.sigmoid(mask_phase) * 1.999 * math.pi
        mask_complex = tf.complex(
            self.A*tf.cos(mask_phase), self.A*tf.sin(mask_phase))
        mask = tf.pad(mask_complex, self.paddings2, 'CONSTANT')
        out = tf.multiply(input, mask)
        return out


class SLM(tf.keras.layers.Layer):
    def __init__(self, name, whole_dim, mask_dim, filling_ratio, phase=None):
        super(SLM, self).__init__(name=name)
        if type(whole_dim) == int:
            whole_dim = (whole_dim, whole_dim)
        if type(mask_dim) == int:
            mask_dim = (mask_dim, mask_dim)
        self.w = self.add_weight("w",
                                 shape=(1,)+mask_dim,
                                 initializer=Constant(
                                     phase) if phase is not None else 'random_normal',
                                 trainable=True,
                                 dtype=tf.float64
                                 )
        self.A = self.add_weight("A", shape=(1,),
                                 initializer="random_normal",
                                 trainable=True,
                                 dtype=tf.float64)
        self.paddings = get_padding(3, whole_dim, mask_dim)
        self.filling_ratio = filling_ratio

    def call(self, input):
        mask_phase = tf.sigmoid(self.w) * 1.999 * math.pi
        mask_complex = tf.complex(
            self.A*tf.cos(mask_phase), self.A*tf.sin(mask_phase))
        mask = tf.pad(mask_complex, self.paddings, 'CONSTANT')
        out = tf.multiply(input, mask)*tf.complex(self.filling_ratio,
                                                  0.0) + input*tf.complex((1-self.filling_ratio)*self.A, 0.0)
        return out


class Scalar(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Scalar, self).__init__(name=name)
        self.A = self.add_weight("A", shape=(1,),
                                 initializer="random_normal",
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, input):
        return input*tf.abs(self.A)


class MyTestPhase(tf.keras.callbacks.Callback):
    def __init__(self, folder, delta_image, delta_label, phase_layer, output_slice=slice(None)):
        super(MyTestPhase, self).__init__()
        self.folder = folder
        self.delta_image = delta_image
        self.delta_label = np.array(delta_label, dtype=np.float32)
        self.phase_layer = phase_layer
        self.output_slice = output_slice

    def on_train_begin(self, logs):
        self.best_epoch = 0
        self.best_train_loss = float("inf")

    def on_epoch_end(self, epoch, logs):
        if logs['loss'] < self.best_train_loss:
            self.best_epoch = epoch+1
            self.best_train_loss = logs['loss']
            delta_output = self.model(self.delta_image)
            plt.subplot(121)
            plt.imshow(delta_output[0, self.output_slice, self.output_slice])
            plt.subplot(122)
            plt.imshow(
                self.delta_label[0, self.output_slice, self.output_slice])
            plt.savefig(self.folder+'/temp_results/%04d.png' % (epoch+1))
            plt.clf()
            w = self.model.layers[self.phase_layer].w[0].numpy()
            with open(self.folder+'/param_%04d.npy' % (epoch+1), 'wb') as f:
                np.save(f, w)
