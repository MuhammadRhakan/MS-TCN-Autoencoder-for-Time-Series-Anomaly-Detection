from keras.layers import Conv1D, Conv1DTranspose, Activation, Dropout, LayerNormalization
from keras.models import Model

import numpy as np
import tensorflow as tf
import keras
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hides INFO and WARNING logs


class Backbone(keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.gelu1 = Activation('gelu')
        self.dropout1 = Dropout(0.2)

        self.conv2 = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate)
        self.gelu2 = Activation('gelu')
        self.dropout2 = Dropout(0.2)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.gelu1(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.gelu2(x)
        return self.dropout2(x, training=training)


class TCNResidualBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv1 = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        self.norm1 = LayerNormalization()
        self.relu1 = Activation('relu')
        self.dropout1 = Dropout(0.2)

        self.conv2 = Conv1DTranspose(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)
        self.norm2 = LayerNormalization()
        self.relu2 = Activation('relu')
        self.dropout2 = Dropout(0.2)

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut = Conv1D(filters=self.filters, kernel_size=1, padding='same')
        else:
            self.shortcut = Activation('linear')
        super().build(input_shape)

    def call(self, X, training=None):
        h = self.conv1(X)
        h = self.norm1(h)
        h = self.relu1(h)
        h = self.dropout1(h, training=training)

        h = self.conv2(h)
        h = self.norm2(h)
        out = self.relu2(h + self.shortcut(X))

        return self.dropout2(out, training=training)


class MultiScaleEncoderBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.size1 = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')

        self.size31 = Backbone(filters=16, kernel_size=3, dilation_rate=1)
        self.size32 = Backbone(filters=32, kernel_size=3, dilation_rate=2)
        self.size33 = Backbone(filters=64, kernel_size=3, dilation_rate=4)

        self.size51 = Backbone(filters=16, kernel_size=5, dilation_rate=1)
        self.size52 = Backbone(filters=32, kernel_size=5, dilation_rate=2)
        self.size53 = Backbone(filters=64, kernel_size=5, dilation_rate=4)

        self.concat_branches = keras.layers.Concatenate(axis=-1)
        self.compress = Conv1D(filters=32, kernel_size=1, padding='same', activation='relu')

    def call(self, inputs):
        s1 = self.size1(inputs)

        s3 = self.size31(inputs)
        s3 = self.size32(s3)
        s3 = self.size33(s3)

        s5 = self.size51(inputs)
        s5 = self.size52(s5)
        s5 = self.size53(s5)

        concat = self.concat_branches([s1, s3, s5])
        return self.compress(concat)


class AutoEncoder(keras.models.Model):
    def __init__(self, features, latent_dim):
        super().__init__()
        self.encoder_blocks = MultiScaleEncoderBlock()
        self.compress = Conv1D(filters=latent_dim, kernel_size=2, strides=2, activation='relu', padding='valid')
        self.expand = Conv1DTranspose(filters=48, kernel_size=2, strides=2, activation='relu', padding='valid')
        self.decoder_blocks = keras.Sequential([
            TCNResidualBlock(filters=32, kernel_size=2, dilation_rate=4),
            TCNResidualBlock(filters=64, kernel_size=2, dilation_rate=2),
            TCNResidualBlock(filters=128, kernel_size=2, dilation_rate=1)
        ])
        self.out = Conv1D(filters=features, kernel_size=1, activation='linear')

    def call(self, inputs):
        encoder = self.encoder_blocks(inputs)
        bottleneck = self.compress(encoder)
        reshape = self.expand(bottleneck)
        decoder = self.decoder_blocks(reshape)
        return self.out(decoder)