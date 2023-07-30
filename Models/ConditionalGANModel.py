import numpy as np
from sklearn.utils import shuffle
import plotly.graph_objects as go
import plotly.io as pio
import os
import keras

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import multiply
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import LeakyReLU
from keras.layers import PReLU
from keras.layers import Add

from keras.models import Sequential
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import Adadelta

DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION = 128
DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM = "Adam"
DEFAULT_CONDITIONAL_GAN_ACTIVATION = "LeakyReLU"
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_CONDITIONAL_GAN_BATCH_SIZE = 32
DEFAULT_CONDITIONAL_GAN_NUMBER_CLASSES = 2
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G = [128, 256, 512]
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D = [512, 256, 128]
DEFAULT_CONDITIONAL_GAN_LOSS = "binary_crossentropy"
DEFAULT_CONDITIONAL_GAN_MOMENTUM = 0.8

DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE = 0.0001
DEFAULT_CONDITIONAL_GAN_ADAM_BETA = 0.9

DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE = 0.001
DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE = 0.95

DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE = 0.001
DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE = 0.95


class ConditionalGAN:
    def __init__(self, latent_dim=DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION, output_shape=None,
                 training_algorithm=DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM,
                 activation_function=DEFAULT_CONDITIONAL_GAN_ACTIVATION,
                 dropout_decay_rate_g=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G,
                 dropout_decay_rate_d=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D,
                 dense_layer_sizes_g=DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G,
                 dense_layer_sizes_d=DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D,
                 batch_size=DEFAULT_CONDITIONAL_GAN_BATCH_SIZE, dataset_type=np.float32, output_dir=None):

        self.latent_dim = latent_dim
        self.out_shape = output_shape
        self.num_classes = DEFAULT_CONDITIONAL_GAN_NUMBER_CLASSES
        self.training_algorithm = training_algorithm
        self.activation_function = activation_function
        self.dropout_decay_rate_g = dropout_decay_rate_g
        self.dropout_decay_rate_d = dropout_decay_rate_d
        self.dense_layer_sizes_g = dense_layer_sizes_g
        self.dense_layer_sizes_d = dense_layer_sizes_d
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.output_dir = output_dir

        self.instance_discriminator = self.discriminator()
        self.instance_discriminator.compile(loss=[DEFAULT_CONDITIONAL_GAN_LOSS], optimizer=self.get_optimizer())
        self.instance_generator = self.generator()
        self.latent_input = Input(shape=(self.latent_dim,))
        self.label_input = Input(shape=(1,))

        generator_instance = self.instance_generator([self.latent_input, self.label_input])
        self.instance_discriminator.trainable = False
        discriminator_flow = self.instance_discriminator([generator_instance, self.label_input])
        self.combined = Model([self.latent_input, self.label_input], discriminator_flow)
        self.combined.compile(loss=[DEFAULT_CONDITIONAL_GAN_LOSS], optimizer=self.get_optimizer())
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    def get_optimizer(self):

        if self.training_algorithm == 'Adam':
            return Adam(DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE,
                        DEFAULT_CONDITIONAL_GAN_ADAM_BETA)

        elif self.training_algorithm == 'RMSprop':
            return RMSprop(DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE,
                           DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE)

        elif self.training_algorithm == 'Adadelta':
            return Adadelta(DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE,
                            DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE)

        else:
            raise ValueError("Algoritmo de treinamento inválido. Use 'Adam', 'RMSprop' ou 'Adadelta'.")

    def add_activation_layer(self, neural_nodel):

        if self.activation_function == 'LeakyReLU':
            neural_nodel = LeakyReLU()(neural_nodel)
        elif self.activation_function == 'ReLU':
            neural_nodel = Activation('relu')(neural_nodel)
        elif self.activation_function == 'PReLU':
            neural_nodel = PReLU()(neural_nodel)

        return neural_nodel

    def generator(self):

        neural_model_inputs = Input(shape=(self.latent_dim,), dtype=self.dataset_type)
        latent_input = Input(shape=(self.latent_dim,))

        label_input = Input(shape=(1,), dtype=self.dataset_type)
        generator_model = Dense(self.dense_layer_sizes_g[0])(neural_model_inputs)
        generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
        generator_model = self.add_activation_layer(generator_model)
        generator_model = BatchNormalization(momentum=DEFAULT_CONDITIONAL_GAN_MOMENTUM)(generator_model)

        for layer_size in self.dense_layer_sizes_g[1:]:
            generator_model = Dense(layer_size)(generator_model)
            generator_model = Dropout(self.dropout_decay_rate_g)(generator_model)
            generator_model = self.add_activation_layer(generator_model)
            generator_model = BatchNormalization(momentum=DEFAULT_CONDITIONAL_GAN_MOMENTUM)(generator_model)

        generator_model = Dense(self.out_shape, activation='sigmoid')(generator_model)
        generator_model = Model(inputs=neural_model_inputs, outputs=generator_model)

        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label_input))
        model_input = Add()([latent_input, label_embedding])
        generator_output_flow = generator_model(model_input)
        v = Model([latent_input, label_input], generator_output_flow, name="Generator")
        v.summary()
        return Model([latent_input, label_input], generator_output_flow, name="Generator")

    def discriminator(self):

        neural_model_input = Input(shape=(self.out_shape,), dtype=self.dataset_type)
        generator_shape_input = Input(shape=(self.out_shape,))
        label_input = Input(shape=(1,), dtype=self.dataset_type)

        discriminator_model = Dense(self.dense_layer_sizes_g[0])(neural_model_input)
        discriminator_model = Dropout(self.dropout_decay_rate_g)(discriminator_model)
        discriminator_model = self.add_activation_layer(discriminator_model)
        discriminator_model = BatchNormalization(momentum=DEFAULT_CONDITIONAL_GAN_MOMENTUM)(discriminator_model)

        for layer_size in self.dense_layer_sizes_g[1:]:
            discriminator_model = Dense(layer_size)(discriminator_model)
            discriminator_model = Dropout(self.dropout_decay_rate_g)(discriminator_model)
            discriminator_model = self.add_activation_layer(discriminator_model)
            discriminator_model = BatchNormalization(momentum=DEFAULT_CONDITIONAL_GAN_MOMENTUM)(discriminator_model)

        discriminator_model = Dense(1, activation='sigmoid')(discriminator_model)
        discriminator_model = Model(inputs=neural_model_input, outputs=discriminator_model)

        label_embedding = Flatten()(Embedding(self.num_classes, self.out_shape)(label_input))

        model_input = Add()([generator_shape_input, label_embedding])
        validity = discriminator_model(model_input)
        v = Model(inputs=[generator_shape_input, label_input], outputs=validity, name="Discriminator")
        v.summary()
        return Model(inputs=[generator_shape_input, label_input], outputs=validity, name="Discriminator")

    def train(self, x_training, y_training, epochs, batch_size=32, sample_interval=100):

        if batch_size is None:
            batch_size = self.batch_size

        generator_loss, discriminator_loss = [], []

        labels_real_data, labels_synthetic_data = np.ones((batch_size, 1)), np.zeros((batch_size, 1))
        
        for epoch in range(epochs):

            idx = np.random.randint(0, x_training.shape[0], batch_size)
            samples_on_batch, labels = x_training[idx], y_training[idx]
            samples_on_batch, labels = shuffle(samples_on_batch, labels)
            random_latent_noise = np.random.normal(0, 0.1, (self.batch_size, self.latent_dim))
            samples_generated = self.instance_generator.predict([random_latent_noise, labels], verbose=0)
            d_loss_real = self.instance_discriminator.train_on_batch([samples_on_batch, labels], labels_real_data)
            d_loss_synthetic = self.instance_discriminator.train_on_batch([samples_generated, labels],
                                                                          labels_synthetic_data)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_synthetic)
            sampled_labels = np.random.randint(0, 2, batch_size).reshape(-1, 1)
            random_latent_noise = np.random.normal(0, 0.1, (self.batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch([random_latent_noise, np.array(sampled_labels, dtype=np.float)],
                                                  labels_real_data)
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)

            if (epoch + 1) % sample_interval == 0:
                print("[%d/%d]\tLoss Discriminator: %.4f\tLoss Generator: %.4f" % (epoch, epochs,
                                                                                   self.disc_loss_tracker.result(),
                                                                                   self.gen_loss_tracker.result()))
            generator_loss.append(self.gen_loss_tracker.result())
            discriminator_loss.append(self.disc_loss_tracker.result())

        if self.output_dir is not None:

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(generator_loss))), y=generator_loss, name='Generator'))
            fig.add_trace(go.Scatter(x=list(range(len(discriminator_loss))), y=discriminator_loss, name='Discriminator'))
            fig.update_layout(title="Perda do Gerador e Discriminador",
                              xaxis_title="iterações",
                              yaxis_title="Perda",
                              legend_title="Legenda")

            pio.write_image(fig, os.path.join(self.output_dir, "curve_training_error.pdf"))
