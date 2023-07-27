import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, classification_report, ConfusionMatrixDisplay
import gc
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.initializers import RandomNormal
import keras.backend as K
import plotly.graph_objects as go
import contextlib
import warnings
import logging


class cGAN():
    def __init__(self, latent_dim, out_shape, training_algorithm='Adam', activation_function='LeakyReLU', dropout_decay_rate_g=0.2, dropout_decay_rate_d=0.4, dense_layer_sizes_g=[128, 256, 512], dense_layer_sizes_d=[512, 256, 128], batch_size=32):
        self.latent_dim = latent_dim
        self.out_shape = out_shape
        self.num_classes = 2
        self.training_algorithm = training_algorithm
        self.activation_function = activation_function
        self.dropout_decay_rate_g = dropout_decay_rate_g
        self.dropout_decay_rate_d = dropout_decay_rate_d
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d
        self.batch_size = batch_size

        # Definindo o otimizador com base no algoritmo de treinamento escolhido
        optimizer = self.get_optimizer()

        # Construindo o discriminador
        self.discriminator = self.discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Construindo o gerador
        self.generator = self.generator()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        gen_samples = self.generator([noise, label])

        # Não treinamos o discriminador ao treinar o gerador
        self.discriminator.trainable = False
        valid = self.discriminator([gen_samples, label])

        # Combinando os dois modelos
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def get_optimizer(self):
        # Método para obter o otimizador com base no algoritmo de treinamento escolhido
        if self.training_algorithm == 'Adam':
            return Adam(0.0002, 0.5)
        elif self.training_algorithm == 'RMSprop':
            return RMSprop(lr=0.0002, decay=6e-8)
        elif self.training_algorithm == 'Adadelta':
            return Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        else:
            raise ValueError("Algoritmo de treinamento inválido. Use 'Adam', 'RMSprop' ou 'Adadelta'.")

    def generator(self):
        # Criação do modelo gerador
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        for layer_size in self.dense_layer_sizes_g:
            model.add(Dense(layer_size, input_dim=self.latent_dim))
            model.add(Dropout(self.dropout_decay_rate_g))

            if self.activation_function == 'LeakyReLU':
                model.add(LeakyReLU(alpha=0.2))
            elif self.activation_function == 'ReLU':
                model.add(Activation('relu'))
            elif self.activation_function == 'PReLU':
                model.add(PReLU())
            else:
                raise ValueError("Função de ativação inválida. Use 'LeakyReLU', 'ReLU' ou 'PReLU'.")
            model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.out_shape, activation='sigmoid'))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int8')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        gen_sample = model(model_input)

        return Model([noise, label], gen_sample, name="Generator")

    def discriminator(self):
        # Criação do modelo discriminador
        init = RandomNormal(mean=0.0, stddev=0.02)
        model = Sequential()

        for layer_size in self.dense_layer_sizes_d:
            model.add(Dense(layer_size, input_dim=self.out_shape, kernel_initializer=init))
            if self.activation_function == 'LeakyReLU':
                model.add(LeakyReLU(alpha=0.2))
            elif self.activation_function == 'ReLU':
                model.add(Activation('relu'))
            elif self.activation_function == 'PReLU':
                model.add(PReLU())
            else:
                raise ValueError("Função de ativação inválida. Use 'LeakyReLU', 'ReLU' ou 'PReLU'.")
            model.add(Dropout(self.dropout_decay_rate_d))

        model.add(Dense(1, activation='sigmoid'))

        gen_sample = Input(shape=(self.out_shape,))
        label = Input(shape=(1,), dtype='int8')
        label_embedding = Flatten()(Embedding(self.num_classes, self.out_shape)(label))

        model_input = multiply([gen_sample, label_embedding])
        validity = model(model_input)

        return Model(inputs=[gen_sample, label], outputs=validity, name="Discriminator")

    def train(self, X_train, y_train, pos_index, neg_index, epochs, sampling=False, batch_size=32, sample_interval=100, plot=True):
        if batch_size is None:
            batch_size = self.batch_size

        # Embora não recomendado, definir perdas como globais ajuda na análise do nosso cGAN fora da classe
        global G_losses
        global D_losses

        G_losses = []
        D_losses = []
        # Verdades básicas adversárias
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # if sample==True --> treina o discriminador com 8 amostras da classe positiva e o resto com classe negativa
            if sampling:
                idx1 = np.random.choice(pos_index, 8)
                idx0 = np.random.choice(neg_index, batch_size - 8)
                idx = np.concatenate((idx1, idx0))
            # if sample!=True --> treina o discriminador usando instâncias aleatórias em lotes de 32
            else:
                idx = np.random.choice(len(y_train), batch_size)
            samples, labels = X_train[idx], y_train[idx]
            samples, labels = shuffle(samples, labels)

            # Amostra de ruído como entrada do gerador
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_samples = self.generator.predict([noise, labels])

            # Alisamento de rótulo
            if epoch < epochs // 1.5:
                valid_smooth = (valid + 0.1) - (np.random.random(valid.shape) * 0.1)
                fake_smooth = (fake - 0.1) + (np.random.random(fake.shape) * 0.1)
            else:
                valid_smooth = valid
                fake_smooth = fake

            # Treinando o discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch([samples, labels], valid_smooth)
            d_loss_fake = self.discriminator.train_on_batch([gen_samples, labels], fake_smooth)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Treinando o Generator
            self.discriminator.trainable = False
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)
            # Treinando o generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            if (epoch + 1) % sample_interval == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch, epochs, d_loss[0], g_loss[0]))
            G_losses.append(g_loss[0])
            D_losses.append(d_loss[0])
            if plot:
                if epoch + 1 == epochs:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(len(G_losses))), y=G_losses, name='G'))
                    fig.add_trace(go.Scatter(x=list(range(len(D_losses))), y=D_losses, name='D'))
                    fig.update_layout(title="Perda do Gerador e Discriminador",
                                      xaxis_title="iterações",
                                      yaxis_title="Perda",
                                      legend_title="Legenda")

                    fig.show()