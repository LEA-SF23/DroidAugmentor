#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

import logging
import os
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers.legacy import Adadelta
import tensorflow as tf

DEFAULT_OPTIMIZER_GENERATOR_LEARNING = 0.0001
DEFAULT_OPTIMIZER_DISCRIMINATOR_LEARNING = 0.0001
DEFAULT_OPTIMIZER_GENERATOR_BETA = 0.5
DEFAULT_OPTIMIZER_DISCRIMINATOR_BETA = 0.5
DEFAULT_LATENT_DIMENSION = 128

DEFAULT_OPTIMIZER_GENERATOR = Adam(DEFAULT_OPTIMIZER_GENERATOR_LEARNING, DEFAULT_OPTIMIZER_GENERATOR_BETA)
DEFAULT_OPTIMIZER_DISCRIMINATOR = Adam(DEFAULT_OPTIMIZER_DISCRIMINATOR_LEARNING, DEFAULT_OPTIMIZER_DISCRIMINATOR_BETA)
DEFAULT_LOSS_GENERATOR = BinaryCrossentropy()
DEFAULT_LOSS_DISCRIMINATOR = BinaryCrossentropy()

DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE = 0.0001
DEFAULT_CONDITIONAL_GAN_ADAM_BETA = 0.5
DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE = 0.001
DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE = 0.5
DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE = 0.001
DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE = 0.5
DEFAULT_FILE_NAME_DISCRIMINATOR = "discriminator_model"
DEFAULT_FILE_NAME_GENERATOR = "generator_model"
DEFAULT_PATH_OUTPUT_MODELS = "models_saved/"


class AdversarialModel(Model):
    def __init__(self, generator_model=None, discriminator_model=None, latent_dimension=DEFAULT_LATENT_DIMENSION,
                 optimizer_generator=DEFAULT_OPTIMIZER_GENERATOR, loss_generator=DEFAULT_LOSS_GENERATOR,
                 optimizer_discriminator=DEFAULT_OPTIMIZER_DISCRIMINATOR, loss_discriminator=DEFAULT_LOSS_DISCRIMINATOR,
                 conditional_gan_adam_learning_rate=DEFAULT_CONDITIONAL_GAN_ADAM_LEARNING_RATE,
                 conditional_gan_adam_beta=DEFAULT_CONDITIONAL_GAN_ADAM_BETA,
                 conditional_gan_rms_prop_learning_rate=DEFAULT_CONDITIONAL_GAN_RMS_PROP_LEARNING_RATE,
                 conditional_gan_rms_prop_decay_rate=DEFAULT_CONDITIONAL_GAN_RMS_PROP_DECAY_RATE,
                 conditional_gan_ada_delta_learning_rate=DEFAULT_CONDITIONAL_GAN_ADA_DELTA_LEARNING_RATE,
                 conditional_gan_ada_delta_decay_rate=DEFAULT_CONDITIONAL_GAN_ADA_DELTA_DECAY_RATE,
                 file_name_discriminator=DEFAULT_FILE_NAME_DISCRIMINATOR,
                 file_name_generator=DEFAULT_FILE_NAME_GENERATOR, models_saved_path=DEFAULT_PATH_OUTPUT_MODELS,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator_model
        self.discriminator = discriminator_model
        self.latent_dimension = latent_dimension
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.loss_generator = loss_generator
        self.loss_discriminator = loss_discriminator

        self.conditional_gan_adam_learning_rate = conditional_gan_adam_learning_rate
        self.conditional_gan_adam_beta = conditional_gan_adam_beta
        self.conditional_gan_rms_prop_learning_rate = conditional_gan_rms_prop_learning_rate
        self.conditional_gan_rms_prop_decay_rate = conditional_gan_rms_prop_decay_rate
        self.conditional_gan_ada_delta_learning_rate = conditional_gan_ada_delta_learning_rate
        self.conditional_gan_ada_delta_decay_rate = conditional_gan_ada_delta_decay_rate
        self.file_name_discriminator = file_name_discriminator
        self.file_name_generator = file_name_generator
        self.models_saved_path = models_saved_path

    def compile(self, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, *args,
                **kwargs):
        super().compile(*args, **kwargs)
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.loss_generator = loss_generator
        self.loss_discriminator = loss_discriminator

    @tf.function
    def train_step(self, batch):

        real_feature, real_samples_label = batch
        batch_size = tf.shape(real_feature)[0]
        real_samples_label = tf.expand_dims(real_samples_label, axis=-1)
        latent_space = tf.random.normal(shape=(batch_size, self.latent_dimension))
        synthetic_feature = self.generator([latent_space, real_samples_label], training=False)

        with tf.GradientTape() as discriminator_gradient:
            label_predicted_real = self.discriminator([real_feature, real_samples_label], training=True)
            label_predicted_synthetic = self.discriminator([synthetic_feature, real_samples_label], training=True)
            label_predicted_all_samples = tf.concat([label_predicted_real, label_predicted_synthetic], axis=0)
            list_all_labels_predicted = [tf.zeros_like(label_predicted_real), tf.ones_like(label_predicted_synthetic)]
            tensor_labels_predicted = tf.concat(list_all_labels_predicted, axis=0)

            smooth_tensor_real_data = 0.15 * tf.random.uniform(tf.shape(label_predicted_real))
            smooth_tensor_synthetic_data = -0.15 * tf.random.uniform(tf.shape(label_predicted_synthetic))
            tensor_labels_predicted += tf.concat([smooth_tensor_real_data, smooth_tensor_synthetic_data], axis=0)
            loss_value = self.loss_discriminator(tensor_labels_predicted, label_predicted_all_samples)

        gradient_tape_loss = discriminator_gradient.gradient(loss_value, self.discriminator.trainable_variables)
        self.optimizer_discriminator.apply_gradients(zip(gradient_tape_loss, self.discriminator.trainable_variables))

        with tf.GradientTape() as generator_gradient:
            latent_space = tf.random.normal(shape=(batch_size, self.latent_dimension))
            synthetic_feature = self.generator([latent_space, real_samples_label], training=True)
            predicted_labels = self.discriminator([synthetic_feature, real_samples_label], training=False)
            total_loss_g = self.loss_generator(tf.zeros_like(predicted_labels), predicted_labels)

        gradient_tape_loss = generator_gradient.gradient(total_loss_g, self.generator.trainable_variables)
        self.optimizer_generator.apply_gradients(zip(gradient_tape_loss, self.generator.trainable_variables))

        return {"loss_d": loss_value, "loss_g": total_loss_g}

    def save_models(self, path_output, k_fold):

        try:

            logging.info("Saving Adversarial Model:")
            path_directory = os.path.join(path_output, self.models_saved_path)
            Path(path_directory).mkdir(parents=True, exist_ok=True)

            discriminator_file_name = self.file_name_discriminator + "_" + str(k_fold)
            generator_file_name = self.file_name_generator + "_" + str(k_fold)

            path_model = os.path.join(path_directory, "fold_" + str(k_fold + 1))
            Path(path_model).mkdir(parents=True, exist_ok=True)

            discriminator_file_name = os.path.join(path_model, discriminator_file_name)
            generator_file_name = os.path.join(path_model, generator_file_name)

            discriminator_model_json = self.discriminator.to_json()

            with open(discriminator_file_name + ".json", "w") as json_file:
                json_file.write(discriminator_model_json)

            self.discriminator.save_weights(discriminator_file_name + ".h5")

            generator_model_json = self.generator.to_json()

            with open(generator_file_name + ".json", "w") as json_file:
                json_file.write(generator_model_json)

            self.generator.save_weights(generator_file_name + ".h5")

            logging.info("  Discriminator output: {}".format(discriminator_file_name))
            logging.info("  Generator output: {}".format(generator_file_name))

        except FileExistsError:

            logging.error("File model exists")
            exit(-1)

    def load_models(self, path_output, k_fold):

        try:

            logging.info("Loading Adversarial Model:")
            path_directory = os.path.join(path_output, self.models_saved_path)

            discriminator_file_name = self.file_name_discriminator + "_" + str(k_fold + 1)
            generator_file_name = self.file_name_generator + "_" + str(k_fold + 1)

            discriminator_file_name = os.path.join(path_directory, discriminator_file_name)
            generator_file_name = os.path.join(path_directory, generator_file_name)

            discriminator_model_json_pointer = open(discriminator_file_name + ".json", 'r')
            discriminator_model_json = discriminator_model_json_pointer.read()
            discriminator_model_json_pointer.close()

            self.discriminator = model_from_json(discriminator_model_json)
            self.discriminator.load_weights(discriminator_file_name + ".h5")

            generator_model_json_pointer = open(generator_file_name + ".json", 'r')
            generator_model_json = generator_model_json_pointer.read()
            generator_model_json_pointer.close()

            self.generator = model_from_json(generator_model_json)
            self.generator.load_weights(generator_file_name + ".h5")

            logging.info("Model loaded: {}".format(discriminator_file_name))
            logging.info("Model loaded: {}".format(generator_file_name))

        except FileNotFoundError:

            logging.error("Forneça um modelo existente e válido")
            exit(-1)

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_latent_dimension(self, latent_dimension):
        self.latent_dimension = latent_dimension

    def set_optimizer_generator(self, optimizer_generator):
        self.optimizer_generator = optimizer_generator

    def set_optimizer_discriminator(self, optimizer_discriminator):
        self.optimizer_discriminator = optimizer_discriminator

    def set_loss_generator(self, loss_generator):
        self.loss_generator = loss_generator

    def set_loss_discriminator(self, loss_discriminator):
        self.loss_discriminator = loss_discriminator

    def get_optimizer(self, training_algorithm, first_arg=None, second_arg=None):

        if training_algorithm == 'Adam':

            if first_arg is None:
                first_arg = self.conditional_gan_adam_learning_rate

            if second_arg is None:
                second_arg = self.conditional_gan_adam_beta

            return Adam(first_arg, second_arg)

        elif training_algorithm == 'RMSprop':

            if first_arg is None:
                first_arg = self.conditional_gan_rms_prop_learning_rate

            if second_arg is None:
                second_arg = self.conditional_gan_rms_prop_decay_rate

            return RMSprop(first_arg, second_arg)

        elif training_algorithm == 'Adadelta':

            if first_arg is None:
                first_arg = self.conditional_gan_ada_delta_learning_rate

            if second_arg is None:
                second_arg = self.conditional_gan_ada_delta_decay_rate

            return Adadelta(first_arg, second_arg)

        else:
            raise ValueError("Algoritmo de treinamento inválido. Use 'Adam', 'RMSprop' ou 'Adadelta'.")
