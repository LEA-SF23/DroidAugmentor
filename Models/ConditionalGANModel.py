#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

import numpy as np

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import Concatenate
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.models import Model


DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION = 128
DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM = "Adam"
DEFAULT_CONDITIONAL_GAN_ACTIVATION = "LeakyReLU"
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_CONDITIONAL_GAN_BATCH_SIZE = 32
DEFAULT_CONDITIONAL_GAN_NUMBER_CLASSES = 2
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G = [128]
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D = [128]
DEFAULT_CONDITIONAL_GAN_LOSS = "binary_crossentropy"
DEFAULT_CONDITIONAL_GAN_MOMENTUM = 0.8
DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_CONDITIONAL_GAN_INITIALIZER_MEAN = 0.0
DEFAULT_CONDITIONAL_GAN_INITIALIZER_DEVIATION = 0.02


class ConditionalGAN:
    def __init__(self, latent_dim=DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION, output_shape=None,
                 activation_function=DEFAULT_CONDITIONAL_GAN_ACTIVATION,
                 initializer_mean=DEFAULT_CONDITIONAL_GAN_INITIALIZER_MEAN,
                 initializer_deviation=DEFAULT_CONDITIONAL_GAN_INITIALIZER_DEVIATION,
                 dropout_decay_rate_g=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D,
                 last_layer_activation=DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER,
                 dense_layer_sizes_g=None, dense_layer_sizes_d=None, dataset_type=np.float32):

        if dense_layer_sizes_d is None:
            dense_layer_sizes_d = DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D

        if dense_layer_sizes_g is None:
            dense_layer_sizes_g = DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G

        self.latent_dim = latent_dim
        self.output_shape = output_shape
        self.activation_function = activation_function
        self.last_layer_activation = last_layer_activation
        self.dropout_decay_rate_g = dropout_decay_rate_g
        self.dropout_decay_rate_d = dropout_decay_rate_d
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d
        self.dataset_type = dataset_type
        self.initializer_mean = initializer_mean
        self.initializer_deviation = initializer_deviation
        self.generator_model_dense = None
        self.discriminator_model_dense = None

    def add_activation_layer(self, neural_nodel):

        if self.activation_function == 'LeakyReLU':
            neural_nodel = LeakyReLU()(neural_nodel)

        elif self.activation_function == 'ReLU':
            neural_nodel = Activation('relu')(neural_nodel)

        elif self.activation_function == 'PReLU':
            neural_nodel = PReLU()(neural_nodel)

        return neural_nodel

    def get_generator(self):

        initialization = RandomNormal(mean=self.initializer_mean, stddev=self.initializer_deviation)
        neural_model_inputs = Input(shape=(self.latent_dim,), dtype=self.dataset_type)
        latent_input = Input(shape=(self.latent_dim,))

        label_input = Input(shape=(1,), dtype=self.dataset_type)

        generator_model = Dense(self.dense_layer_sizes_g[0], kernel_initializer=initialization)(neural_model_inputs)
        #generator_model = BatchNormalization(momentum=0.8)(generator_model)
        generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
        generator_model = self.add_activation_layer(generator_model)

        for layer_size in self.dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size, kernel_initializer=initialization)(generator_model)
            #generator_model = BatchNormalization(momentum=0.8)(generator_model)
            generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
            generator_model = self.add_activation_layer(generator_model)

        generator_model = Dense(self.output_shape, self.last_layer_activation, kernel_initializer=initialization)(
            generator_model)
        generator_model = Model(neural_model_inputs, generator_model, name="Dense_Generator")
        self.generator_model_dense = generator_model
        concatenate_output = Concatenate()([latent_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self.latent_dim)(label_embedding)
        generator_output_flow = generator_model(model_input)

        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def get_discriminator(self):

        neural_model_input = Input(shape=(self.output_shape,), dtype=self.dataset_type)
        discriminator_shape_input = Input(shape=(self.output_shape,))
        label_input = Input(shape=(1,), dtype=self.dataset_type)

        discriminator_model = Dense(self.dense_layer_sizes_d[0])(neural_model_input)
        discriminator_model = Dropout(self.dropout_decay_rate_d)(discriminator_model)
        #discriminator_model = BatchNormalization(momentum=0.8)(discriminator_model)
        discriminator_model = self.add_activation_layer(discriminator_model)

        for layer_size in self.dense_layer_sizes_d[1:]:

            discriminator_model = Dense(layer_size)(discriminator_model)
            #discriminator_model = BatchNormalization(momentum=0.8)(discriminator_model)
            discriminator_model = Dropout(self.dropout_decay_rate_d)(discriminator_model)
            discriminator_model = self.add_activation_layer(discriminator_model)

        discriminator_model = Dense(1, self.last_layer_activation)(discriminator_model)
        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_model, name="Dense_Discriminator")
        self.discriminator_model_dense = discriminator_model
        concatenate_output = Concatenate()([discriminator_shape_input, label_input])
        label_embedding = Flatten()(concatenate_output)
        model_input = Dense(self.output_shape)(label_embedding)
        validity = discriminator_model(model_input)
        return Model(inputs=[discriminator_shape_input, label_input], outputs=validity, name="Discriminator")

    def get_dense_generator_model(self):
        return self.generator_model_dense

    def get_dense_discriminator_model(self):
        return self.discriminator_model_dense

    def set_latent_dimension(self, latent_dimension):
        self.latent_dim = latent_dimension

    def set_output_shape(self, output_shape):
        self.output_shape = output_shape

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def set_last_layer_activation(self, last_layer_activation):
        self.last_layer_activation = last_layer_activation

    def set_dropout_decay_rate_generator(self, dropout_decay_rate_generator):
        self.dropout_decay_rate_g = dropout_decay_rate_generator

    def set_dropout_decay_rate_discriminator(self, dropout_decay_rate_discriminator):
        self.dropout_decay_rate_d = dropout_decay_rate_discriminator

    def set_dense_layer_sizes_generator(self, dense_layer_sizes_generator):
        self.dense_layer_sizes_g = dense_layer_sizes_generator

    def set_dense_layer_sizes_discriminator(self, dense_layer_sizes_discriminator):
        self.dense_layer_sizes_d = dense_layer_sizes_discriminator

    def set_dataset_type(self, dataset_type):
        self.dataset_type = dataset_type

    def set_initializer_mean(self, initializer_mean):
        self.initializer_mean = initializer_mean

    def set_initializer_deviation(self, initializer_deviation):
        self.initializer_deviation = initializer_deviation
