#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']


try:

    import json
    import os
    import sys
    import logging
    import datetime
    import argparse
    import warnings
    import math
    from logging.handlers import RotatingFileHandler
    import mlflow
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from aim import Image, Distribution,Run
    from pathlib import Path
    from tensorflow.keras.losses import BinaryCrossentropy
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    
    from Models.ConditionalGANModel import ConditionalGAN
    from Models.AdversarialModel import AdversarialModel
    from Models.Classifiers import Classifiers

    from Tools.tools import PlotConfusionMatrix
    from Tools.tools import PlotRegressiveMetrics
    from Tools.tools import PlotCurveLoss
    from Tools.tools import ProbabilisticMetrics
    from Tools.tools import PlotClassificationMetrics
    from Tools.tools import DEFAULT_COLOR_NAME 
    USE_AIM=False
    USE_MLFLOW=False
   # if USE_AIM:
    #import aim
     # from aim import Run


except ImportError as error:

    print(error)
    print()
    print("1. (optional) Setup a virtual environment: ")
    print("  python3 - m venv ~/Python3env/DroidAugmentor ")
    print("  source ~/Python3env/DroidAugmentor/bin/activate ")
    print()
    print("2. Install requirements:")
    print("  pip3 install --upgrade pip")
    print("  pip3 install -r requirements.txt ")
    print()
    sys.exit(-1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")
    warnings.filterwarnings("ignore", message="Variables are collinear")

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"

DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES = 2000
DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES = 2000
DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 100
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
DEFAULT_ADVERSARIAL_ACTIVATION = "LeakyReLU"  # ['LeakyReLU', 'ReLU', 'PReLU']
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.02
DEFAULT_ADVERSARIAL_BATCH_SIZE = 32
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [128]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [128]
DEFAULT_ADVERSARIAL_RANDOM_LATENT_MEAN_DISTRIBUTION = 0.0
DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION = 1.0

DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_SAVE_MODELS = True
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "training_curve"
DEFAULT_CLASSIFIER_LIST = ["RandomForest",  
                           "KNN",
                           "DecisionTree"]  #"AdaBoost", "SupportVectorMachine", "QuadraticDiscriminant", "NaiveBayes","Perceptron"

DEFAULT_VERBOSE_LIST = {logging.INFO: 2, logging.DEBUG: 1, logging.WARNING: 2,
                        logging.FATAL: 0, logging.ERROR: 0}

LOGGING_FILE_NAME = "logging.log"
aim_run=None

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))


# Define a custom argument type for a list of integers
def list_of_strs(arg):
    return list(map(str, arg.split(',')))


def generate_samples(instance_model, number_instances, latent_dimension, label_class, verbose_level,
                     latent_mean_distribution, latent_stander_deviation):

    if np.ceil(label_class) == 1:
        label_samples_generated = np.ones(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))
    else:
        label_samples_generated = np.zeros(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))

    latent_noise = np.random.normal(latent_mean_distribution, latent_stander_deviation,
                                    (number_instances, latent_dimension))
    generated_samples = instance_model.generator.predict([latent_noise, label_samples_generated], verbose=verbose_level)
    generated_samples = np.rint(generated_samples)

    return generated_samples, label_samples_generated


def comparative_data(fold, x_synthetic, real_data):
    instance_metrics = ProbabilisticMetrics()
    synthetic_mean_squared_error = instance_metrics.get_mean_squared_error(real_data, x_synthetic)
    synthetic_cosine_similarity = instance_metrics.get_cosine_similarity(real_data, x_synthetic)
    synthetic_kl_divergence = instance_metrics.get_mean_squared_error(real_data, x_synthetic)
    synthetic_maximum_mean_discrepancy = instance_metrics.get_mean_squared_error(real_data, x_synthetic)
    logging.info(f"Similarity Metrics")
    logging.info(f"  Synthetic Fold {fold + 1} - Mean Squared Error: " + str(synthetic_mean_squared_error))
    logging.info(f"  Synthetic Fold {fold + 1} - Cosine Similarity: " + str(synthetic_cosine_similarity))
    logging.info(f"  Synthetic Fold {fold + 1} - KL Divergence: " + str(synthetic_kl_divergence))
    logging.info(f"  Synthetic Fold {fold + 1} - Maximum Mean Discrepancy: " + str(synthetic_maximum_mean_discrepancy))
    logging.info("")

    return [synthetic_mean_squared_error, synthetic_cosine_similarity, synthetic_kl_divergence,
            synthetic_maximum_mean_discrepancy]


def evaluate_synthetic_data(list_classifiers, x_synthetic, y_synthetic, fold, k, generate_confusion_matrix,
                            output_dir, classifier_type, out_label, path_confusion_matrix, verbose_level):
    instance_metrics = ProbabilisticMetrics()
    accuracy_synthetic_list, precision_synthetic_list, recall_synthetic_list, f1_score_synthetic_list = [], [], [], []

    logging.info(f"Synthetic Fold {fold + 1}/{k} results\n")

    for index, classifier_model in enumerate(list_classifiers):

        if classifier_type[index] == "Perceptron":
            y_predicted_synthetic = classifier_model.predict(x_synthetic, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_synthetic = np.rint(np.squeeze(y_predicted_synthetic, axis=1))
        else:

            y_predicted_synthetic = classifier_model.predict(x_synthetic)

        confusion_matrix_synthetic = confusion_matrix(y_synthetic, y_predicted_synthetic)
        accuracy_synthetic = instance_metrics.get_accuracy(y_synthetic, y_predicted_synthetic)
        precision_synthetic = instance_metrics.get_precision(y_synthetic, y_predicted_synthetic)
        recall_synthetic = instance_metrics.get_recall(y_synthetic, y_predicted_synthetic)
        f1_score_synthetic = instance_metrics.get_f1_score(y_synthetic, y_predicted_synthetic)
        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   Synthetic Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_synthetic)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     Synthetic Fold {fold + 1} - Accuracy: " + str(accuracy_synthetic))
        logging.info(f"     Synthetic Fold {fold + 1} - Precision: " + str(precision_synthetic))
        logging.info(f"     Synthetic Fold {fold + 1} - Recall: " + str(recall_synthetic))
        logging.info(f"     Synthetic Fold {fold + 1} - F1 Score: " + str(f1_score_synthetic) + "\n")
        values={'accuracy_synthetic':accuracy_synthetic,'precision_synthetic':precision_synthetic,'r_recall_synthetic':recall_synthetic,'f1_score':f1_score_synthetic}

        if generate_confusion_matrix:
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_synthetic, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_Synthetic_{classifier_type[index]}_k{fold + 1}.jpg')
            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image,name= f'CM_Synthetic_{classifier_type[index]}_k{fold + 1}')
            if USE_MLFLOW:
                  mlflow.log_metrics(values)
                  mlflow.log_artifact(matrix_file, 'images')
        accuracy_synthetic_list.append(accuracy_synthetic)
        precision_synthetic_list.append(precision_synthetic)
        recall_synthetic_list.append(recall_synthetic)
        f1_score_synthetic_list.append(f1_score_synthetic)

    return [accuracy_synthetic_list, precision_synthetic_list, recall_synthetic_list, f1_score_synthetic_list]


def evaluate_real_data(list_classifiers, x_real, y_real, fold, k, generate_confusion_matrix, output_dir,
                       classifier_type, out_label, path_confusion_matrix, verbose_level):
    logging.info(f"Real Fold {fold + 1}/{k} results")
    instance_metrics = ProbabilisticMetrics()
    accuracy_real_list, precision_real_list, recall_real_list, f1_real_list = [], [], [], []

    for index, classifier_model in enumerate(list_classifiers):

        if classifier_type[index] == "Perceptron":
            y_predicted_real = classifier_model.predict(x_real, verbose=DEFAULT_VERBOSE_LIST[verbose_level])
            y_predicted_real = np.rint(np.squeeze(y_predicted_real, axis=1))
        else:
            y_predicted_real = classifier_model.predict(x_real)

        y_predicted_real = y_predicted_real.astype(int)
        y_sample_real = y_real.astype(int)
        confusion_matrix_real = confusion_matrix(y_sample_real, y_predicted_real)

        accuracy_real = instance_metrics.get_accuracy(y_sample_real, y_predicted_real)
        precision_real = instance_metrics.get_precision(y_sample_real, y_predicted_real)
        recall_real = instance_metrics.get_recall(y_sample_real, y_predicted_real)
        f1_real = instance_metrics.get_f1_score(y_sample_real, y_predicted_real)

        logging.info(f" Classifier Model: {classifier_type[index]}")
        logging.info(f"   Real Fold {fold + 1} - Confusion Matrix:")
        logging.info(confusion_matrix_real)
        logging.info(f"\n   Classifier Metrics:")
        logging.info(f"     Real Fold {fold + 1} - Accuracy: " + str(accuracy_real))
        logging.info(f"     Real Fold {fold + 1} - Precision: " + str(precision_real))
        logging.info(f"     Real Fold {fold + 1} - Recall: " + str(recall_real))
        logging.info(f"     Real Fold {fold + 1} - F1 Score: " + str(f1_real) + "\n")
        logging.info("")
        values={'accuracy_real':accuracy_real,'precision_real':precision_real,'recall_real':recall_real,'f1_real':f1_real}
        if generate_confusion_matrix:
            plt.figure()
            selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME[(fold + 2) % len(DEFAULT_COLOR_NAME)])
            confusion_matrix_instance = PlotConfusionMatrix()
            confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_real, out_label, selected_color_map)
            Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
            matrix_file = os.path.join(output_dir, path_confusion_matrix,
                                       f'CM_Real_{classifier_type[index]}_k{fold + 1}.jpg')
           
            if USE_AIM:
               aim_run.track(values)
               plt.savefig(matrix_file, bbox_inches='tight')
               aim_image = Image(matrix_file)
               aim_run.track(value=aim_image, name=f'CM_Real_{classifier_type[index]}_k{fold + 1}')
     #                 context={'classifier_type': classifier_type[index]})
            if USE_MLFLOW:
                  mlflow.log_metrics(values)
                  mlflow.log_artifact(matrix_file, 'images')

        accuracy_real_list.append(accuracy_real)
        precision_real_list.append(precision_real)
        recall_real_list.append(recall_real)
        f1_real_list.append(f1_real)

    return [accuracy_real_list, precision_real_list, recall_real_list, f1_real_list]


def show_and_export_results(synthetic_accuracies, synthetic_precisions, synthetic_recalls, synthetic_f1_scores,
                            real_accuracies, real_precisions, real_recalls, real_f1_scores, list_mean_squared_error,
                            list_cosine_similarity, list_kl_divergence, list_max_mean_discrepancy, classifier_type,
                            output_dir, title_output_label):
    plot_classifier_metrics = PlotClassificationMetrics()
    plot_regressive_metrics = PlotRegressiveMetrics()

    for index in range(len(classifier_type)):
        logging.info("Overall Synthetic Results: Classifier {}\n".format(classifier_type[index]))
        logging.info("  Synthetic List of Accuracies: {}".format(synthetic_accuracies[index]))
        logging.info("  Synthetic List of Precisions: {}".format(synthetic_precisions[index]))
        logging.info("  Synthetic List of Recalls: {}".format(synthetic_recalls[index]))
        logging.info("  Synthetic List of F1-scores: {}".format(synthetic_f1_scores[index]))
        logging.info("  Synthetic Mean Accuracy: {}".format(np.mean(synthetic_accuracies[index])))
        logging.info("  Synthetic Mean Precision: {}".format(np.mean(synthetic_precisions[index])))
        logging.info("  Synthetic Mean Recall: {}".format(np.mean(synthetic_recalls[index])))
        logging.info("  Synthetic Mean F1 Score: {}".format(np.mean(synthetic_f1_scores[index])))
        logging.info("  Synthetic Standard Deviation of Accuracy: {}".format(np.std(synthetic_accuracies[index])))
        logging.info("  Synthetic Standard Deviation of Precision: {}".format(np.std(synthetic_precisions[index])))
        logging.info("  Synthetic Standard Deviation of Recall: {}".format(np.std(synthetic_recalls[index])))
        logging.info("  Synthetic Standard Deviation of F1 Score: {}\n".format(np.std(synthetic_f1_scores[index])))

        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_Synthetic.pdf')

        plot_classifier_metrics.plot_classifier_metrics(classifier_type[index], synthetic_accuracies[index],
                                                        synthetic_precisions[index], synthetic_recalls[index],
                                                        synthetic_f1_scores[index], plot_filename,
                                                        f'{title_output_label}_SYNTHETIC')


        values = {'accuracy_mean': np.mean(synthetic_accuracies[index]),
                  'precision_mean': np.mean(synthetic_precisions[index]),
                  'recall_mean': np.mean(synthetic_recalls[index]),
                  'f1_mean': np.mean(synthetic_f1_scores[index]),
                  'accuracy_std': np.std(synthetic_accuracies[index]),
                  'precision_std': np.std(synthetic_precisions[index]),
                  'recall_std': np.std(synthetic_recalls[index]),
                  'f1_std': np.std(synthetic_f1_scores[index]),
                  }
        if USE_AIM:
           aim_run.track(values, context={'data': 'Synthetic', 'classifier_type': classifier_type[index]})
           d = Distribution(synthetic_accuracies[index])
           aim_run.track(d, name='accuracy_dist', step=0, context={'data': 'Synthetic', 'classifier_type': classifier_type[index]})
        if USE_MLFLOW:
            mlflow.log_metrics(values,step=index)
            #Distribution(synthetic_accuracies[index])
            mlflow.log_metric('accuracy_dist',synthetic_accuracies[index],step=index)

        logging.info("Overall Real Results: {}\n".format(classifier_type[index]))
        logging.info("  Real List of Accuracies: {}".format(real_accuracies[index]))
        logging.info("  Real List of Precisions: {}".format(real_precisions[index]))
        logging.info("  Real List of Recalls: {}".format(real_recalls[index]))
        logging.info("  Real List of F1-scores: {}".format(real_f1_scores[index]))
        logging.info("  Real Mean Accuracy: {}".format(np.mean(real_accuracies[index])))
        logging.info("  Real Mean Precision: {}".format(np.mean(real_precisions[index])))
        logging.info("  Real Mean Recall: {}".format(np.mean(real_recalls[index])))
        logging.info("  Real Mean F1 Score: {}".format(np.mean(real_f1_scores[index])))
        logging.info("  Real Standard Deviation of Accuracy: {}".format(np.std(real_accuracies[index])))
        logging.info("  Real Standard Deviation of Precision: {}".format(np.std(real_precisions[index])))
        logging.info("  Real Standard Deviation of Recall: {}".format(np.std(real_recalls[index])))
        logging.info("  Real Standard Deviation of F1 Score: {}\n".format(np.std(real_f1_scores[index])))

        plot_filename = os.path.join(output_dir, f'{classifier_type[index]}_Real.pdf')

        plot_classifier_metrics.plot_classifier_metrics(classifier_type[index], real_accuracies[index],
                                                        real_precisions[index], real_recalls[index],
                                                        real_f1_scores[index], plot_filename,
                                                        f'{title_output_label}_REAL')
        values = {'accuracy_mean': np.mean(real_accuracies[index]),
                      'precision_mean': np.mean(real_precisions[index]),
                      'recall_mean': np.mean(real_recalls[index]),
                      'f1_mean': np.mean(real_f1_scores[index]),
                      'accuracy_std': np.std(real_accuracies[index]),
                      'precision_std': np.std(real_precisions[index]),
                      'recall_std': np.std(real_recalls[index]),
                      'f1_std': np.std(real_f1_scores[index]),
                      }
        if USE_AIM:
           aim_run.track(values,context={'data': 'Real','classifir_type': classifier_type[index]})
           d = Distribution(real_accuracies[index])
           aim_run.track(d, name='accuracy_dist', step=0, context={'data': 'Real', 'classifier_type': classifier_type[index]})
        if USE_MLFLOW:
            mlflow.log_metrics(values,step=index)
            d = Distribution(real_accuracies[index])

            mlflow.log_metric('accuracy_dist',real_accuracies[index],step=index)


    comparative_metrics = ['Mean Squared Error', 'Cosine Similarity', 'KL divergence ', 'Max Mean Discrepancy']
    comparative_lists = [list_mean_squared_error, list_cosine_similarity, list_kl_divergence, list_max_mean_discrepancy]
    logging.info(f"Comparative Metrics:")
    for metric, comparative_list in zip(comparative_metrics, comparative_lists):
        logging.info("\t{}".format(metric))
        logging.info("\t\t{} - List     : {}".format(metric, comparative_list))
        logging.info("\t\t{} - Mean     : {}".format(metric, np.mean(comparative_list)))
        logging.info("\t\t{} - Std. Dev.: {}\n".format(metric, np.std(comparative_list)))
        

    plot_filename = os.path.join(output_dir, f'Comparison_Real_Synthetic.jpg')

    plot_regressive_metrics.plot_regressive_metrics(list_mean_squared_error, list_cosine_similarity,
                                                    list_kl_divergence, list_max_mean_discrepancy, plot_filename,
                                                    f'{title_output_label}')
    if USE_AIM:
       aim_image = Image(plot_filename)
       aim_run.track(value=aim_image, name=f'Comparison_Real_Synthetic',context={'classifier_type': classifier_type[index]})
    if USE_MLFLOW:
       mlflow.log_artifact(plot_filename,'images')
def get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean, initializer_deviation,
                          dropout_decay_rate_g, dropout_decay_rate_d, last_layer_activation, dense_layer_sizes_g,
                          dense_layer_sizes_d, dataset_type, training_algorithm, latent_mean_distribution,
                          latent_stander_deviation):
    instance_models = ConditionalGAN(latent_dim, input_data_shape, activation_function, initializer_mean,
                                     initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                     last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d, dataset_type)

    generator_model = instance_models.get_generator()
    discriminator_model = instance_models.get_discriminator()

    adversarial_model = AdversarialModel(generator_model, discriminator_model, latent_dimension=latent_dim,
                                         latent_mean_distribution=latent_mean_distribution,
                                         latent_stander_deviation=latent_stander_deviation)

    optimizer_generator = adversarial_model.get_optimizer(training_algorithm)
    optimizer_discriminator = adversarial_model.get_optimizer(training_algorithm)
    loss_generator = BinaryCrossentropy()
    loss_discriminator = BinaryCrossentropy()

    adversarial_model.compile(optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator)

    return adversarial_model


def show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level):
    show_model_instance = ConditionalGAN(latent_dim, input_data_shape, activation_function, initializer_mean,
                                         initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                         last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                         dataset_type)

    if verbose_level == logging.INFO:
        logging.info("Model Architecture")
        logging.info("  Model Generator")
        show_model_instance.get_generator().summary()
        logging.info("  Model Discriminator")
        show_model_instance.get_discriminator().summary()

    if verbose_level == logging.DEBUG:
        logging.info("Model Architecture")
        logging.info("  Model Generator")
        show_model_instance.get_generator().summary()
        logging.info("  Dense Structure")
        show_model_instance.get_dense_generator_model().summary()
        logging.info("  Model Discriminator")
        show_model_instance.get_discriminator().summary()
        logging.info("  Dense Structure")
        show_model_instance.get_dense_discriminator_model().summary()
        logging.info("")

def run_experiment(dataset, input_data_shape, k, classifier_list, output_dir, batch_size, training_algorithm,
                   number_epochs, latent_dim, activation_function, dropout_decay_rate_g, dropout_decay_rate_d,
                   dense_layer_sizes_g=None, dense_layer_sizes_d=None, dataset_type=None, title_output=None,
                   initializer_mean=None, initializer_deviation=None,
                   last_layer_activation=DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER, save_models=False,
                   path_confusion_matrix=None, path_curve_loss=None, verbose_level=None,
                   latent_mean_distribution=None, latent_stander_deviation=None, num_samples_class_malware=None, num_samples_class_benign=None):

    show_model(latent_dim, input_data_shape, activation_function, initializer_mean,
               initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
               last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
               dataset_type, verbose_level)

    stratified = StratifiedKFold(n_splits=k, shuffle=True)

    list_accuracy, list_precision, list_recall, list_f1_score = [], [], [], []
    list_mean_squared_error, list_cosine_similarity, list_kl_divergence, list_maximum_mean_discrepancy = [], [], [], []
    list_real_accuracy, list_real_precision, list_real_recall, list_real_f1_score = [], [], [], []

    for i, (train_index, test_index) in enumerate(stratified.split(dataset.iloc[:, :-1], dataset.iloc[:, -1])):

        adversarial_model = get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean,
                                                  initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                                  last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                                  dataset_type, training_algorithm, latent_mean_distribution,
                                                  latent_stander_deviation)





        instance_classifier = Classifiers()

        x_training = np.array(dataset.iloc[train_index, :-1].values, dtype=dataset_type)
        x_test = np.array(dataset.iloc[test_index, :-1].values, dtype=dataset_type)
        y_training = np.array(dataset.iloc[train_index, -1].values, dtype=dataset_type)
        y_test = np.array(dataset.iloc[test_index, -1].values, dtype=dataset_type)
        x_training = x_training[0:int(len(x_training) - (len(x_training) % batch_size)), :]
        y_training = y_training[0:int(len(y_training) - (len(y_training) % batch_size))]

        logging.info(f" Starting training ADVERSARIAL MODEL:\n")
        training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size,
                                                 verbose=DEFAULT_VERBOSE_LIST[verbose_level])
        logging.info(f"     Finished training\n")


        if save_models:
            adversarial_model.save_models(output_dir, i)

        generator_loss_list = training_history.history['loss_g']
        discriminator_loss_list = training_history.history['loss_d']
        plot_loss_curve_instance = PlotCurveLoss()
        plot_loss_curve_instance.plot_training_loss_curve(generator_loss_list, discriminator_loss_list, output_dir, i,
                                                          path_curve_loss)

        number_samples_true = len([positional_label for positional_label in y_test.tolist() if positional_label == 1])
        number_samples_false = len([positional_label for positional_label in y_test.tolist() if positional_label == 0])



        # Calcular o número desejado de amostras sintéticas para cada classe
        num_samples_true_desired = num_samples_class_malware
        num_samples_false_desired = num_samples_class_benign

        # Gerar dados sintéticos para a classe verdadeira (label 1)
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, num_samples_true_desired, latent_dim,
                                                              1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)

        # Gerar dados sintéticos para a classe falsa (label 0)
        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, num_samples_false_desired,
                                                                latent_dim,
                                                                0, verbose_level, latent_mean_distribution,
                                                                latent_stander_deviation)

        # Juntar os dados sintéticos gerados para ambas as classes
        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), axis=0)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic), axis=0))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)

        # Converter para DataFrame e adicionar os nomes das colunas
        synthetic_columns = dataset.columns[:-1]
        df_synthetic = pd.DataFrame(data=x_synthetic_samples, columns=synthetic_columns)
        df_synthetic['class'] = y_synthetic_samples

        if i + 1 == k:
            # Salvar dados sintéticos em um arquivo CSV
            synthetic_filename = f'synthetic_data_fold_{i + 1}.csv'
            synthetic_filepath = os.path.join(output_dir, synthetic_filename)
            df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)
            # Salvar dados sintéticos em um arquivo CSV
            # synthetic_filename = f'synthetic_data_fold_{i + 1}.csv'
            # synthetic_filepath = os.path.join(output_dir, synthetic_filename)
            # df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)



        # Generate synthetic samples with the desired number for each class
        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, num_samples_true_desired, latent_dim,
                                                            1, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)

        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, num_samples_false_desired, latent_dim,
                                                             0, verbose_level, latent_mean_distribution,
                                                              latent_stander_deviation)

        # Ensure the desired number of samples for each class
        x_true_synthetic = x_true_synthetic[:number_samples_true]
        y_true_synthetic = y_true_synthetic[:number_samples_true]

        x_false_synthetic = x_false_synthetic[:number_samples_false]
        y_false_synthetic = y_false_synthetic[:number_samples_false]

        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), dtype=dataset_type)
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic)))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)



        list_classifiers = instance_classifier.get_trained_classifiers(classifier_list, x_training, y_training,
                                                                       dataset_type, verbose_level, input_data_shape)

        evaluation_results_synthetic_data = evaluate_synthetic_data(list_classifiers, x_synthetic_samples,
                                                                    y_synthetic_samples, i, k, True,
                                                                    output_dir, classifier_list, title_output,
                                                                    path_confusion_matrix, verbose_level)

        evaluation_results_real_data = evaluate_real_data(list_classifiers, x_test, y_test, i, k,
                                                          True, output_dir, classifier_list,
                                                          title_output, path_confusion_matrix, verbose_level)

        comparative_metrics = comparative_data(i, x_synthetic_samples, x_test)

        list_accuracy.append(evaluation_results_synthetic_data[0])
        list_precision.append(evaluation_results_synthetic_data[1])
        list_recall.append(evaluation_results_synthetic_data[2])
        list_f1_score.append(evaluation_results_synthetic_data[3])

        list_real_accuracy.append(evaluation_results_real_data[0])
        list_real_precision.append(evaluation_results_real_data[1])
        list_real_recall.append(evaluation_results_real_data[2])
        list_real_f1_score.append(evaluation_results_real_data[3])

        list_mean_squared_error.append(comparative_metrics[0])
        list_cosine_similarity.append(comparative_metrics[1])
        list_kl_divergence.append(comparative_metrics[2])
        list_maximum_mean_discrepancy.append(comparative_metrics[3])

    show_and_export_results(list_accuracy, list_precision, list_recall, list_f1_score, list_real_accuracy,
                            list_real_precision, list_real_recall, list_real_f1_score, list_mean_squared_error,
                            list_cosine_similarity, list_kl_divergence, list_maximum_mean_discrepancy,
                            classifier_list, output_dir, title_output)


def initial_step(initial_arguments, dataset_type):
    
    file_args = os.path.join(initial_arguments.output_dir, 'commandline_args.txt')

    with open(file_args, 'w') as f:
        json.dump(initial_arguments.__dict__, f, indent=2)

    dataset_file_loaded = pd.read_csv(initial_arguments.input_dataset, dtype=dataset_type)
    dataset_file_loaded = dataset_file_loaded.dropna()
    dataset_input_shape = dataset_file_loaded.shape[1] - 1

    input_dataset = os.path.basename(initial_arguments.input_dataset)
    dataset_labels = f'Dataset: {input_dataset}'
    return dataset_file_loaded, dataset_input_shape, dataset_labels


def show_all_settings(arg_parsers):
    logging.info("Command:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
    logging.info("Settings:")
    lengths = [len(x) for x in vars(arg_parsers).keys()]
    max_length = max(lengths)

    for k, v in sorted(vars(arg_parsers).items()):
        settings_parser = "\t"
        settings_parser += k.ljust(max_length, " ")
        settings_parser += " : {}".format(v)
        logging.info(settings_parser)

    logging.info("")


def create_argparse():

    parser = argparse.ArgumentParser(description='Run the experiment with cGAN and classifiers')

    parser.add_argument('-i', '--input_dataset', type=str, required=True,
                        help='Arquivo do dataset de entrada (Formato CSV)')

    parser.add_argument('-c', '--classifier', type=list_of_strs, default=DEFAULT_CLASSIFIER_LIST,
                        help='Classificador (ou lista de classificadores separada por ,) padrão:{}.'.format(
                            DEFAULT_CLASSIFIER_LIST))

    parser.add_argument('-o', '--output_dir', type=str,
                        default=f'out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Diretório para gravação dos arquivos de saída.')

    parser.add_argument('--data_type', type=str, default=DEFAULT_DATA_TYPE,
                        choices=['int8', 'float16', 'float32'],
                        help='Tipo de dado para representar as características das amostras.')

    parser.add_argument('--num_samples_class_malware', type=int,
                        default=DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES,
                        help='Número de amostras da Classe 1 (maligno).')

    parser.add_argument('--num_samples_class_benign', type=int,
                        default=DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES,
                        help='Número de amostras da Classe 0 (benigno).')

    parser.add_argument('--number_epochs', type=int,
                        default=DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN,
                        help='Número de épocas (iterações de treinamento).')

    parser.add_argument('--k_fold', type=int,
                        default=DEFAULT_NUMBER_STRATIFICATION_FOLD,
                        help='Número de folds para validação cruzada.')

    parser.add_argument('--initializer_mean', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_MEAN,
                        help='Valor central da distribuição gaussiana do inicializador.')

    parser.add_argument('--initializer_deviation', type=float,
                        default=DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION,
                        help='Desvio padrão da distribuição gaussiana do inicializador.')

    parser.add_argument("--latent_dimension", type=int,
                        default=DEFAULT_ADVERSARIAL_LATENT_DIMENSION,
                        help="Dimensão do espaço latente para treinamento cGAN")

    # TODO testar adequadamente antes de disponibilizar 'RMSprop', 'Adadelta'
    parser.add_argument("--training_algorithm", type=str,
                        default=DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM,
                        help="Algoritmo de treinamento para cGAN.",
                        choices=['Adam'])

    parser.add_argument("--activation_function",
                        type=str, default=DEFAULT_ADVERSARIAL_ACTIVATION,
                        help="Função de ativação da cGAN.",
                        choices=['LeakyReLU', 'ReLU', 'PReLU'])

    parser.add_argument("--dropout_decay_rate_g",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G,
                        help="Taxa de decaimento do dropout do gerador da cGAN")

    parser.add_argument("--dropout_decay_rate_d",
                        type=float, default=DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D,
                        help="Taxa de decaimento do dropout do discriminador da cGAN")

    parser.add_argument("--dense_layer_sizes_g", type=list_of_ints, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G,
                        help=" Valor das camadas densas do gerador")

    parser.add_argument("--dense_layer_sizes_d", type=list_of_ints, nargs='+',
                        default=DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D,
                        help="valor das camadas densas do discriminador")

    parser.add_argument('--batch_size', type=int,
                        default=DEFAULT_ADVERSARIAL_BATCH_SIZE,
                        choices=[16, 32, 64],
                        help='Tamanho do lote da cGAN.')

    parser.add_argument("--verbosity", type=int,
                        help='Verbosity (Default {})'.format(DEFAULT_VERBOSITY),
                        default=DEFAULT_VERBOSITY)

    parser.add_argument("--save_models", type=bool,
                        help='Salvar modelos treinados (Default {})'.format(DEFAULT_SAVE_MODELS),
                        default=DEFAULT_SAVE_MODELS)

    parser.add_argument("--path_confusion_matrix", type=str,
                        help='Diretório de saída das matrizes de confusão',
                        default=DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX)

    parser.add_argument("--path_curve_loss", type=str,
                        help='Diretório de saída dos gráficos de curva de treinamento',
                        default=DEFAULT_OUTPUT_PATH_TRAINING_CURVE)

    parser.add_argument("--latent_mean_distribution", type=float,
                        help='Média da distribuição do ruído aleatório de entrada',
                        default=DEFAULT_ADVERSARIAL_RANDOM_LATENT_MEAN_DISTRIBUTION)

    parser.add_argument("--latent_stander_deviation", type=float,
                        help='Desvio padrão do ruído aleatório de entrada',
                        default=DEFAULT_ADVERSARIAL_RANDOM_LATENT_STANDER_DEVIATION)
    parser.add_argument('-a','--use_aim',type=bool,help="Uso ou não da ferramenta aim para monitoramento",default=False) 
    
    parser.add_argument('-ml','--use_mlflow',type=bool,help="Uso ou não da ferramenta mlflow para monitoramento",default=False) 
    return parser.parse_args()


if __name__ == "__main__":



    arguments = create_argparse()

    logging_format = '%(asctime)s\t***\t%(message)s'

    # configura o mecanismo de logging
    if arguments.verbosity == logging.DEBUG:
        # mostra mais detalhes
        logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s'

    Path(arguments.output_dir).mkdir(parents=True, exist_ok=True)
    logging_filename = os.path.join(arguments.output_dir, LOGGING_FILE_NAME)

    # formatter = logging.Formatter(logging_format, datefmt=TIME_FORMAT, level=arguments.verbosity)
    logging.basicConfig(format=logging_format, level=arguments.verbosity)

    # Add file rotating handler, with level DEBUG
    rotatingFileHandler = RotatingFileHandler(filename=logging_filename, maxBytes=100000, backupCount=5)
    rotatingFileHandler.setLevel(arguments.verbosity)
    rotatingFileHandler.setFormatter(logging.Formatter(logging_format))
    logging.getLogger().addHandler(rotatingFileHandler)
    
    show_all_settings(arguments)
    

    time_start_campaign = datetime.datetime.now()

    if arguments.data_type == 'int8':
        data_type = np.int8

    elif arguments.data_type == 'float16':
        data_type = np.float16

    else:
        data_type = np.float32

    if arguments.dense_layer_sizes_g != DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G:
        arguments.dense_layer_sizes_g = arguments.dense_layer_sizes_g[0]

    if arguments.dense_layer_sizes_d != DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D:
        arguments.dense_layer_sizes_d = arguments.dense_layer_sizes_d[0]

    if arguments.classifier != DEFAULT_CLASSIFIER_LIST:
        arguments.classifier = arguments.classifier[0]
    if arguments.use_aim == True:
        USE_AIM= True
    if arguments.use_mlflow:
         USE_MLFLOW= True
    if USE_AIM:
        output_dir = arguments.output_dir
        experiment_name= output_dir.split('/')[-1]
        aim_run=Run(experiment=experiment_name)
        # args = arguments
    if USE_MLFLOW:
       mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
       mlflow.set_experiment("droid_augmentor")
       mlflow.start_run()
    dataset_file, output_shape, output_label = initial_step(arguments, data_type)

    run_experiment(dataset_file, output_shape,
                   arguments.k_fold,
                   arguments.classifier, arguments.output_dir, batch_size=arguments.batch_size,
                   training_algorithm=arguments.training_algorithm, number_epochs=arguments.number_epochs,
                   latent_dim=arguments.latent_dimension, activation_function=arguments.activation_function,
                   dropout_decay_rate_g=arguments.dropout_decay_rate_g,
                   dropout_decay_rate_d=arguments.dropout_decay_rate_d,
                   dense_layer_sizes_g=arguments.dense_layer_sizes_g, dense_layer_sizes_d=arguments.dense_layer_sizes_d,
                   dataset_type=data_type, title_output=output_label, initializer_mean=arguments.initializer_mean,
                   initializer_deviation=arguments.initializer_deviation, save_models=arguments.save_models,
                   path_confusion_matrix=arguments.path_confusion_matrix, path_curve_loss=arguments.path_curve_loss,
                   verbose_level=arguments.verbosity, latent_mean_distribution=arguments.latent_mean_distribution,
                   latent_stander_deviation=arguments.latent_stander_deviation, num_samples_class_malware=arguments.num_samples_class_malware, num_samples_class_benign=arguments.num_samples_class_benign )

    time_end_campaign = datetime.datetime.now()
    logging.info("\t Evaluation duration: {}".format(time_end_campaign - time_start_campaign))
