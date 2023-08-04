#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

import logging
import numpy as np
from tensorflow import keras
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Models.PerceptronModel import PerceptronMultilayer

DEFAULT_RANDOM_FOREST_NUMBER_ESTIMATORS = 100
DEFAULT_RANDOM_FOREST_MAX_DEPTH = None
DEFAULT_RANDOM_FOREST_MAX_LEAF_NODES = None

DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION = 1.0
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL = "rbf"
DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE = 3
DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA = "scale"

DEFAULT_KNN_NUMBER_NEIGHBORS = 5
DEFAULT_KNN_WEIGHTS = "uniform"
DEFAULT_KNN_ALGORITHM = "auto"
DEFAULT_KNN_LEAF_SIZE = 30
DEFAULT_KNN_METRIC = "minkowski"

DEFAULT_GAUSSIAN_PROCESS_KERNEL = None
DEFAULT_GAUSSIAN_PROCESS_MAX_ITERATIONS = 20
DEFAULT_GAUSSIAN_PROCESS_OPTIMIZER = "fmin_l_bfgs_b"

DEFAULT_DECISION_TREE_CRITERION = "gini"
DEFAULT_DECISION_TREE_MAX_DEPTH = None
DEFAULT_DECISION_TREE_MAX_FEATURE = None
DEFAULT_DECISION_TREE_MAX_LEAF = None

DEFAULT_ADA_BOOST_ESTIMATOR = None
DEFAULT_ADA_BOOST_NUMBER_ESTIMATORS = 50
DEFAULT_ADA_BOOST_LEARNING_RATE = 1.0
DEFAULT_ADA_BOOST_ALGORITHM = "SAMME.R"

DEFAULT_NAIVE_BAYES_PRIORS = None
DEFAULT_NAIVE_BAYES_VARIATION_SMOOTHING = 1e-09

DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS = None
DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION = 0.0
DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD = 0.0001

DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_PERCEPTRON_LAYER_ACTIVATION = keras.activations.swish
DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION = "sigmoid"
DEFAULT_PERCEPTRON_NUMBER_EPOCHS = 1

DEFAULT_CLASSIFIER_LIST = ["RandomForest", "SupportVectorMachine", "KNN",
                           "GaussianPrecess", "DecisionTree", "AdaBoost",
                           "NaiveBayes", "QuadraticDiscriminant", "Perceptron"]
DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}


class Classifiers:

    def __init__(self, random_forest_number_estimators=DEFAULT_RANDOM_FOREST_NUMBER_ESTIMATORS,
                 random_forest_max_depth=DEFAULT_RANDOM_FOREST_MAX_DEPTH,
                 max_leaf_nodes=DEFAULT_RANDOM_FOREST_MAX_LEAF_NODES, knn_number_neighbors=DEFAULT_KNN_NUMBER_NEIGHBORS,
                 knn_weights=DEFAULT_KNN_WEIGHTS, knn_leaf_size=DEFAULT_KNN_LEAF_SIZE, knn_metric=DEFAULT_KNN_METRIC,
                 knn_algorithm=DEFAULT_KNN_ALGORITHM, support_vector_machine_gamma=DEFAULT_SUPPORT_VECTOR_MACHINE_GAMMA,
                 support_vector_machine_normalization=DEFAULT_SUPPORT_VECTOR_MACHINE_REGULARIZATION,
                 support_vector_machine_kernel=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL,
                 support_vector_machine_kernel_degree=DEFAULT_SUPPORT_VECTOR_MACHINE_KERNEL_DEGREE,
                 gaussian_process_kernel=DEFAULT_GAUSSIAN_PROCESS_KERNEL,
                 gaussian_process_max_iterations=DEFAULT_GAUSSIAN_PROCESS_MAX_ITERATIONS,
                 gaussian_process_optimizer=DEFAULT_GAUSSIAN_PROCESS_OPTIMIZER,
                 decision_tree_criterion=DEFAULT_DECISION_TREE_CRITERION,
                 decision_tree_max_depth=DEFAULT_DECISION_TREE_MAX_DEPTH,
                 decision_tree_max_feature=DEFAULT_DECISION_TREE_MAX_FEATURE,
                 decision_tree_max_leaf=DEFAULT_DECISION_TREE_MAX_LEAF, ada_boost_estimator=DEFAULT_ADA_BOOST_ESTIMATOR,
                 ada_boost_number_estimators=DEFAULT_ADA_BOOST_NUMBER_ESTIMATORS,
                 ada_boost_learning_rate=DEFAULT_ADA_BOOST_LEARNING_RATE,
                 ada_boost_algorithm=DEFAULT_ADA_BOOST_ALGORITHM, naive_bayes_priors=DEFAULT_NAIVE_BAYES_PRIORS,
                 naive_bayes_variation_smoothing=DEFAULT_NAIVE_BAYES_VARIATION_SMOOTHING,
                 quadratic_discriminant_analysis_priors=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_PRIORS,
                 quadratic_discriminant_analysis_regularization=DEFAULT_QUADRATIC_DISCRIMINANT_ANALYSIS_REGULARIZATION,
                 quadratic_discriminant_threshold=DEFAULT_QUADRATIC_DISCRIMINANT_THRESHOLD,
                 perceptron_training_algorithm=DEFAULT_PERCEPTRON_TRAINING_ALGORITHM,
                 perceptron_training_loss=DEFAULT_PERCEPTRON_LOSS, perceptron_training_metric=None,
                 perceptron_layer_activation=DEFAULT_PERCEPTRON_LAYER_ACTIVATION,
                 perceptron_last_layer_activation=DEFAULT_PERCEPTRON_LAST_LAYER_ACTIVATION,
                 perceptron_dropout_decay_rate=DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE,
                 perceptron_number_epochs=DEFAULT_PERCEPTRON_NUMBER_EPOCHS,
                 perceptron_layers_settings=None):

        self.random_forest_number_estimators = random_forest_number_estimators
        self.random_forest_max_depth = random_forest_max_depth
        self.max_leaf_nodes = max_leaf_nodes

        self.knn_number_neighbors = knn_number_neighbors
        self.knn_weights = knn_weights
        self.knn_leaf_size = knn_leaf_size
        self.knn_metric = knn_metric
        self.knn_algorithm = knn_algorithm

        self.support_vector_machine_normalization = support_vector_machine_normalization
        self.support_vector_machine_kernel = support_vector_machine_kernel
        self.support_vector_machine_kernel_degree = support_vector_machine_kernel_degree
        self.support_vector_machine_gamma = support_vector_machine_gamma

        self.gaussian_process_kernel = gaussian_process_kernel
        self.gaussian_process_max_iterations = gaussian_process_max_iterations
        self.gaussian_process_optimizer = gaussian_process_optimizer

        self.decision_tree_criterion = decision_tree_criterion
        self.decision_tree_max_depth = decision_tree_max_depth
        self.decision_tree_max_feature = decision_tree_max_feature
        self.decision_tree_max_leaf = decision_tree_max_leaf

        self.ada_boost_estimator = ada_boost_estimator
        self.ada_boost_number_estimators = ada_boost_number_estimators
        self.ada_boost_learning_rate = ada_boost_learning_rate
        self.ada_boost_algorithm = ada_boost_algorithm

        self.naive_bayes_priors = naive_bayes_priors
        self.naive_bayes_variation_smoothing = naive_bayes_variation_smoothing

        self.quadratic_discriminant_analysis_priors = quadratic_discriminant_analysis_priors
        self.quadratic_discriminant_analysis_regularization = quadratic_discriminant_analysis_regularization
        self.quadratic_discriminant_threshold = quadratic_discriminant_threshold

        self.perceptron_training_algorithm = perceptron_training_algorithm
        self.perceptron_training_loss = perceptron_training_loss
        self.perceptron_layer_activation = perceptron_layer_activation
        self.perceptron_last_layer_activation = perceptron_last_layer_activation
        self.perceptron_dropout_decay_rate = perceptron_dropout_decay_rate
        self.perceptron_number_epochs = perceptron_number_epochs

        if perceptron_training_metric is None:
            self.perceptron_training_metric = DEFAULT_PERCEPTRON_METRIC
        if perceptron_layers_settings is None:
            self.perceptron_layers_settings = DEFAULT_PERCEPTRON_LAYERS_SETTINGS

    def __get_instance_random_forest(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: RANDOM FOREST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        classifier = RandomForestClassifier(n_estimators=self.random_forest_number_estimators,
                                            max_depth=self.random_forest_max_depth, max_leaf_nodes=self.max_leaf_nodes)
        classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return classifier

    def __get_instance_k_neighbors_classifier(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: K-NEIGHBORS NEAREST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = KNeighborsClassifier(n_neighbors=self.knn_number_neighbors, weights=self.knn_weights,
                                                algorithm=self.knn_algorithm, leaf_size=self.knn_leaf_size,
                                                metric=self.knn_metric)
        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_support_vector_machine(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: SUPPORT VECTOR MACHINE")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = SVC(C=self.support_vector_machine_normalization, kernel=self.support_vector_machine_kernel,
                               degree=self.support_vector_machine_kernel_degree,
                               gamma=self.support_vector_machine_gamma)

        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_gaussian_process(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: GAUSSIAN PROCESS")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = GaussianProcessClassifier(kernel=self.gaussian_process_kernel,
                                                     optimizer=self.gaussian_process_optimizer,
                                                     max_iter_predict=self.gaussian_process_max_iterations)

        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_decision_tree(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: DECISION TREE")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = DecisionTreeClassifier(criterion=self.decision_tree_criterion,
                                                  max_depth=self.decision_tree_max_depth,
                                                  max_features=self.decision_tree_max_feature,
                                                  max_leaf_nodes=self.decision_tree_max_leaf)

        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_ada_boost(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: ADA BOOST")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = AdaBoostClassifier(algorithm=self.ada_boost_algorithm,
                                              n_estimators=self.ada_boost_number_estimators,
                                              learning_rate=self.ada_boost_learning_rate)

        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_naive_bayes(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: NAIVE BAYES")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)

        model_classifier = GaussianNB(priors=self.naive_bayes_priors,
                                      var_smoothing=self.naive_bayes_variation_smoothing)

        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_quadratic_discriminant(self, x_samples_training, y_samples_training, dataset_type):
        logging.info("    Starting training classifier: QUADRATIC DISCRIMINANT ANALYSIS")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        model_classifier = QuadraticDiscriminantAnalysis(priors=self.quadratic_discriminant_analysis_priors,
                                                         reg_param=self.quadratic_discriminant_analysis_regularization,
                                                         tol=self.quadratic_discriminant_threshold)
        model_classifier.fit(x_samples_training, y_samples_training)
        logging.info("\r    Finished training\n")

        return model_classifier

    def __get_instance_perceptron(self, x_samples_training, y_samples_training, dataset_type, verbose_level,
                                  input_dataset_shape):
        logging.info("    Starting training classifier: MULTILAYER PERCEPTRON")

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        instance_classifier = PerceptronMultilayer(self.perceptron_layers_settings, self.perceptron_training_metric,
                                                   self.perceptron_training_loss, self.perceptron_training_algorithm,
                                                   dataset_type, self.perceptron_layer_activation,
                                                   self.perceptron_last_layer_activation,
                                                   self.perceptron_dropout_decay_rate)
        model_classifier = instance_classifier.get_model(input_dataset_shape)
        model_classifier.fit(x_samples_training, y_samples_training, epochs=self.perceptron_number_epochs,
                             verbose=DEFAULT_VERBOSE_LIST[verbose_level])
        logging.info("\r    Finished training\n")

        return model_classifier

    def get_trained_classifiers(self, classifiers_list, x_samples_training, y_samples_training, dataset_type,
                                verbose_level, input_dataset_shape):

        logging.info("\nStarting training classifier\n")
        list_instance_classifiers = []

        for classifier_algorithm in classifiers_list:

            if classifier_algorithm == DEFAULT_CLASSIFIER_LIST[0]:

                list_instance_classifiers.append(self.__get_instance_random_forest(x_samples_training,
                                                                                   y_samples_training,
                                                                                   dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[1]:

                list_instance_classifiers.append(self.__get_instance_support_vector_machine(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[2]:

                list_instance_classifiers.append(self.__get_instance_k_neighbors_classifier(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[3]:
                list_instance_classifiers.append(self.__get_instance_gaussian_process(x_samples_training,
                                                                                      y_samples_training,
                                                                                      dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[4]:

                list_instance_classifiers.append(self.__get_instance_decision_tree(x_samples_training,
                                                                                   y_samples_training,
                                                                                   dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[5]:

                list_instance_classifiers.append(self.__get_instance_ada_boost(x_samples_training,
                                                                               y_samples_training,
                                                                               dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[6]:

                list_instance_classifiers.append(self.__get_instance_naive_bayes(x_samples_training,
                                                                                 y_samples_training,
                                                                                 dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[7]:

                list_instance_classifiers.append(self.__get_instance_quadratic_discriminant(x_samples_training,
                                                                                            y_samples_training,
                                                                                            dataset_type))

            elif classifier_algorithm == DEFAULT_CLASSIFIER_LIST[8]:

                list_instance_classifiers.append(self.__get_instance_perceptron(x_samples_training,
                                                                                y_samples_training,
                                                                                dataset_type,
                                                                                verbose_level,
                                                                                input_dataset_shape))

        return list_instance_classifiers

    def set_random_forest_number_estimators(self, random_forest_number_estimators):
        self.random_forest_number_estimators = random_forest_number_estimators

    def set_random_forest_max_depth(self, random_forest_max_depth):
        self.random_forest_max_depth = random_forest_max_depth

    def max_leaf_nodes(self, max_leaf_nodes):
        self.max_leaf_nodes = max_leaf_nodes

    def set_knn_number_neighbors(self, knn_number_neighbors):
        self.knn_number_neighbors = knn_number_neighbors

    def set_knn_weights(self, knn_weights):
        self.knn_weights = knn_weights

    def set_knn_leaf_size(self, knn_leaf_size):
        self.knn_leaf_size = knn_leaf_size

    def set_knn_metric(self, knn_metric):
        self.knn_metric = knn_metric

    def set_knn_algorithm(self, knn_algorithm):
        self.knn_algorithm = knn_algorithm

    def set_support_vector_machine_normalization(self, support_vector_machine_normalization):
        self.support_vector_machine_normalization = support_vector_machine_normalization

    def set_support_vector_machine_kernel(self, support_vector_machine_kernel):
        self.support_vector_machine_kernel = support_vector_machine_kernel

    def set_support_vector_machine_kernel_degree(self, support_vector_machine_kernel_degree):
        self.support_vector_machine_kernel_degree = support_vector_machine_kernel_degree

    def set_support_vector_machine_gamma(self, support_vector_machine_gamma):
        self.support_vector_machine_gamma = support_vector_machine_gamma

    def set_gaussian_process_kernel(self, gaussian_process_kernel):
        self.gaussian_process_kernel = gaussian_process_kernel

    def set_gaussian_process_max_iterations(self, gaussian_process_max_iterations):
        self.gaussian_process_max_iterations = gaussian_process_max_iterations

    def set_gaussian_process_optimizer(self, gaussian_process_optimizer):
        self.gaussian_process_optimizer = gaussian_process_optimizer

    def set_decision_tree_criterion(self, decision_tree_criterion):
        self.decision_tree_criterion = decision_tree_criterion

    def set_decision_tree_max_depth(self, decision_tree_max_depth):
        self.decision_tree_max_depth = decision_tree_max_depth

    def set_decision_tree_max_feature(self, decision_tree_max_feature):
        self.decision_tree_max_feature = decision_tree_max_feature

    def set_decision_tree_max_leaf(self, decision_tree_max_leaf):
        self.decision_tree_max_leaf = decision_tree_max_leaf

    def set_ada_boost_estimator(self, ada_boost_estimator):
        self.ada_boost_estimator = ada_boost_estimator

    def set_ada_boost_number_estimators(self, ada_boost_number_estimators):
        self.ada_boost_number_estimators = ada_boost_number_estimators

    def set_ada_boost_learning_rate(self, ada_boost_learning_rate):
        self.ada_boost_learning_rate = ada_boost_learning_rate

    def set_ada_boost_algorithm(self, ada_boost_algorithm):
        self.ada_boost_algorithm = ada_boost_algorithm

    def set_naive_bayes_priors(self, naive_bayes_priors):
        self.naive_bayes_priors = naive_bayes_priors

    def set_naive_bayes_variation_smoothing(self, naive_bayes_variation_smoothing):
        self.naive_bayes_variation_smoothing = naive_bayes_variation_smoothing

    def set_quadratic_discriminant_analysis_priors(self, quadratic_discriminant_analysis_priors):
        self.quadratic_discriminant_analysis_priors = quadratic_discriminant_analysis_priors

    def set_quadratic_discriminant_analysis_regularization(self, quadratic_discriminant_analysis_regularization):
        self.quadratic_discriminant_analysis_regularization = quadratic_discriminant_analysis_regularization

    def set_quadratic_discriminant_threshold(self, quadratic_discriminant_threshold):
        self.quadratic_discriminant_threshold = quadratic_discriminant_threshold

    def set_perceptron_training_algorithm(self, perceptron_training_algorithm):
        self.perceptron_training_algorithm = perceptron_training_algorithm

    def set_perceptron_training_loss(self, perceptron_training_loss):
        self.perceptron_training_loss = perceptron_training_loss

    def set_perceptron_layer_activation(self, perceptron_layer_activation):
        self.perceptron_layer_activation = perceptron_layer_activation


a = Classifiers()
print(a.random_forest_number_estimators)