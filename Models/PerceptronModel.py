#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from tensorflow import keras
import numpy

DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_PERCEPTRON_LAYER_ACTIVATION = keras.activations.swish
DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION = "sigmoid"
DEFAULT_PERCEPTRON_DATA_TYPE = numpy.float32


class PerceptronMultilayer:
    def __init__(self, layers_settings=None, training_metric=None, training_loss=DEFAULT_PERCEPTRON_LOSS,
                 training_algorithm=DEFAULT_PERCEPTRON_TRAINING_ALGORITHM, data_type=DEFAULT_PERCEPTRON_DATA_TYPE,
                 layer_activation=DEFAULT_PERCEPTRON_LAYER_ACTIVATION,
                 last_layer_activation=DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION,
                 dropout_decay_rate=DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE):

        self.training_algorithm = training_algorithm
        self.training_loss = training_loss
        self.data_type = data_type
        self.layer_activation = layer_activation
        self.last_layer_activation = last_layer_activation
        self.dropout_decay_rate = dropout_decay_rate

        if training_metric is None:
            training_metric = DEFAULT_PERCEPTRON_METRIC

        if layers_settings is None:
            layers_settings = DEFAULT_PERCEPTRON_LAYERS_SETTINGS

        self.training_metric = training_metric
        self.layers_settings = layers_settings

    def get_model(self, input_shape):

        input_layer = Input(shape=input_shape, dtype=self.data_type)

        dense_layer = Dense(self.layers_settings[0], self.layer_activation)(input_layer)

        for number_neurons in self.layers_settings[1:]:
            dense_layer = Dense(number_neurons, self.layer_activation)(dense_layer)
            dense_layer = Dropout(self.dropout_decay_rate)(dense_layer)

        dense_layer = Dense(1, self.last_layer_activation)(dense_layer)
        neural_network_model = Model(input_layer, dense_layer)
        neural_network_model.compile(self.training_algorithm, self.training_loss, self.training_metric)

        return neural_network_model

    def set_training_algorithm(self, training_algorithm):

        self.training_algorithm = training_algorithm

    def set_training_loss(self, training_loss):

        self.training_loss = training_loss

    def set_data_type(self, data_type):

        self.data_type = data_type

    def set_layer_activation(self, layer_activation):

        self.layer_activation = layer_activation

    def set_last_layer_activation(self, last_layer_activation):

        self.last_layer_activation = last_layer_activation

    def set_dropout_decay_rate(self, dropout_decay_rate):

        self.dropout_decay_rate = dropout_decay_rate
