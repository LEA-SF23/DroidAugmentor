#!/usr/bin/python3
# -*- coding: utf-8 -*-


try:

    import json
    import os
    import warnings
    import logging
    import numpy as np
    import pandas as pd
    import datetime
    import argparse
    import itertools
    import statistics
    import sys

    from tensorflow import keras
    from tensorflow.keras.optimizers.legacy import Adam
    from tensorflow.keras.losses import BinaryCrossentropy
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    from Models.ConditionalGANModel import ConditionalGAN
    from Models.AdversarialModel import AdversarialModel
    from Models.PerceptronModel import PerceptronMultilayer

    from Tools.tools import PlotConfusionMatrix
    from Tools.tools import DEFAULT_COLOR_NAME_MAP
    from Tools.tools import DEFAULT_PLOT_BAR_COLOR_MAP
    from Tools.tools import DEFAULT_PLOT_BAR_METRICS_LABELS

    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    from pathlib import Path

    import plotly.graph_objects as go
    import plotly.io as pio
    import matplotlib.pyplot as plt

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

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")

DEFAULT_VERBOSITY = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'
DEFAULT_DATA_TYPE = "float32"

DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES = 2000
DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES = 2000
DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 1
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_ADVERSARIAL_LATENT_DIMENSION = 128
DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM = "Adam"
DEFAULT_ADVERSARIAL_ACTIVATION = "LeakyReLU"
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_ADVERSARIAL_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_ADVERSARIAL_INITIALIZER_MEAN = 0.0
DEFAULT_ADVERSARIAL_INITIALIZER_DEVIATION = 0.02
DEFAULT_ADVERSARIAL_BATCH_SIZE = 32
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_G = [128]
DEFAULT_ADVERSARIAL_DENSE_LAYERS_SETTINGS_D = [128]

DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER = "sigmoid"
DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]
DEFAULT_SAVE_MODELS = True
DEFAULT_OUTPUT_PATH_CONFUSION_MATRIX = "confusion_matrix"
DEFAULT_OUTPUT_PATH_TRAINING_CURVE = "Training_curve"


# Define a custom argument type for a list of integers
def list_of_ints(arg):
    return list(map(int, arg.split(',')))[0]
    #return map(int, arg.split(','))[0]
    return [int(x) for x in arg.split(',')]
    
def create_plot_classifier_metrics(classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename, title):

    values = [accuracies, precisions, recalls, f1_scores]

    fig = go.Figure()

    for metric, metric_values, color in zip(DEFAULT_PLOT_BAR_METRICS_LABELS, values, DEFAULT_PLOT_BAR_COLOR_MAP):
        metric_mean = statistics.mean(metric_values)
        metric_std = statistics.stdev(metric_values)
        fig.add_trace(go.Bar(x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                             error_y=dict(type='constant', value=metric_std, visible=True), width=0.2))
        fig.add_annotation(x=metric, y=metric_mean + metric_std, xref="x", yref="y", text=f' {metric_std:.4f}',
                           showarrow=False, font=dict(color='black', size=12), xanchor='center', yanchor='bottom')

    y_label_dictionary = dict(title=f'Média {len(accuracies)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
                              gridcolor='black', gridwidth=.05)
    fig.update_layout(barmode='group', title=title, yaxis=y_label_dictionary,
                      xaxis=dict(title=f'Desempenho com {classifier_type}'), showlegend=False, plot_bgcolor='white')

    pio.write_image(fig, plot_filename)


def generate_samples(instance_model, number_instances, latent_dimension, label_class):

    if np.ceil(label_class) == 1:

        label_samples_generated = np.ones(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))

    else:

        label_samples_generated = np.zeros(number_instances, dtype=np.float32)
        label_samples_generated = label_samples_generated.reshape((number_instances, 1))

    random_latent_noise = np.random.normal(0, 1, (number_instances, latent_dimension))
    generated_samples = instance_model.generator.predict([random_latent_noise, label_samples_generated])
    generated_samples = np.rint(generated_samples)

    return generated_samples, label_samples_generated


def get_classifier(classifier_type, x_samples_training, y_samples_training, output_shape, dataset_type):

    if classifier_type == 'knn':
        # TODO: Verificar sobre o KNN
        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(x_samples_training, y_samples_training)

    elif classifier_type == 'perceptron':

        classifier = PerceptronMultilayer().get_model(output_shape)
        classifier.fit(x_samples_training, y_samples_training, epochs=30)

    elif classifier_type == 'random_forest':
        # TODO: Verificar sobre o Random Forest
        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(x_samples_training, y_samples_training)

    elif classifier_type == 'svm':
        # TODO: Verificar sobre o SVM
        parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        svm_classifier = SVC()
        classifier = GridSearchCV(svm_classifier, parameters, cv=3)  # TODO: Verificar cv igual a 3 ou 2?
        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier.fit(x_samples_training, y_samples_training)

    else:
        raise ValueError("Invalid classifier type. Use 'knn', 'perceptron', 'random_forest' or 'svm'.")

    return classifier


def evaluate_synthetic_data(classifier, x_synthetic_samples, y_synthetic_samples, fold, k, generate_confusion_matrix,
                            output_dir, classifier_type, out_label, path_confusion_matrix):

    y_predicted_synthetic = np.rint(classifier.predict(x_synthetic_samples))
    confusion_matrix_synthetic = confusion_matrix(y_synthetic_samples, y_predicted_synthetic)
    accuracy_synthetic = accuracy_score(y_synthetic_samples, y_predicted_synthetic)
    precision_synthetic = precision_score(y_synthetic_samples, y_predicted_synthetic)
    recall_synthetic = recall_score(y_synthetic_samples, y_predicted_synthetic)
    f1_synthetic = f1_score(y_synthetic_samples, y_predicted_synthetic)

    logging.info(f"Synthetic Fold {fold + 1}/{k} results")
    logging.info(f"Synthetic Fold {fold + 1} - Confusion Matrix:")
    logging.info(confusion_matrix_synthetic)
    logging.info(f"Synthetic Fold {fold + 1} - Accuracy: " + str(accuracy_synthetic))
    logging.info(f"Synthetic Fold {fold + 1} - Precision: " + str(precision_synthetic))
    logging.info(f"Synthetic Fold {fold + 1} - Recall: " + str(recall_synthetic))
    logging.info(f"Synthetic Fold {fold + 1} - F1 Score: " + str(f1_synthetic))
    logging.info("---")
    logging.info("")

    if generate_confusion_matrix:

        plt.figure()
        selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME_MAP[(fold + 2) % len(DEFAULT_COLOR_NAME_MAP)])
        confusion_matrix_instance = PlotConfusionMatrix()
        confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_synthetic, out_label, selected_color_map)
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        matrix_file = os.path.join(output_dir, path_confusion_matrix, f'cm_synthetic_{classifier_type}_k{fold + 1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')

    return [accuracy_synthetic, precision_synthetic, recall_synthetic, f1_synthetic]


def evaluate_real_data(classifier, x_real_samples, y_real_samples, fold, k, generate_confusion_matrix, output_dir,
                       classifier_type, out_label, path_confusion_matrix):
    y_predicted_real = classifier.predict(x_real_samples)

    y_predicted_real = y_predicted_real.astype(int)
    y_sample_real = y_real_samples.astype(int)

    confusion_matrix_real = confusion_matrix(y_sample_real, y_predicted_real)

    accuracy_real = accuracy_score(y_sample_real, y_predicted_real)
    precision_real = precision_score(y_sample_real, y_predicted_real)
    recall_real = recall_score(y_sample_real, y_predicted_real)
    f1_real = f1_score(y_sample_real, y_predicted_real)

    logging.info(f"Real Fold {fold + 1}/{k} results")
    logging.info(f"Real Fold {fold + 1} results")
    logging.info(f"Real Fold {fold + 1} - Confusion Matrix:")
    logging.info(confusion_matrix_real)
    logging.info(f"Real Fold {fold + 1} - Accuracy: " + str(accuracy_real))
    logging.info(f"Real Fold {fold + 1} - Precision: " + str(precision_real))
    logging.info(f"Real Fold {fold + 1} - Recall: " + str(recall_real))
    logging.info(f"Real Fold {fold + 1} - F1 Score: " + str(f1_real))
    logging.info("---")

    if generate_confusion_matrix:
        plt.figure()
        selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME_MAP[(fold + 2) % len(DEFAULT_COLOR_NAME_MAP)])
        confusion_matrix_instance = PlotConfusionMatrix()
        confusion_matrix_instance.plot_confusion_matrix(confusion_matrix_real, out_label, selected_color_map)
        Path(os.path.join(output_dir, path_confusion_matrix)).mkdir(parents=True, exist_ok=True)
        matrix_file = os.path.join(output_dir, path_confusion_matrix, f'cm_real_{classifier_type}_k{fold + 1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')

    return [accuracy_real, precision_real, recall_real, f1_real]


def plot_training_loss_curve(generator_loss, discriminator_loss, output_dir, k_fold, path_curve_loss):
    if output_dir is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(generator_loss))), y=generator_loss, name='Generator'))
        fig.add_trace(go.Scatter(x=list(range(len(discriminator_loss))), y=discriminator_loss, name='Discriminator'))
        fig.update_layout(title="Perda do Gerador e Discriminador", xaxis_title="iterações", yaxis_title="Perda",
                          legend_title="Legenda")
        Path(os.path.join(output_dir, path_curve_loss)).mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, os.path.join(output_dir, path_curve_loss,  "curve_training_error_k_{}.pdf".format(k_fold)))


def show_and_export_results(synthetic_accuracies, synthetic_precisions, synthetic_recalls, synthetic_f1_scores,
                            real_accuracies, real_precisions, real_recalls, real_f1_scores, classifier_type,
                            output_dir, title_output_label):
    logging.info(f"Overall Synthetic Results:")

    logging.info("  Synthetic List of Accuracies: {}".format(synthetic_accuracies))
    logging.info("  Synthetic List of Precisions: {}".format(synthetic_precisions))
    logging.info("  Synthetic List of Recalls: {}".format(synthetic_recalls))
    logging.info("  Synthetic List of F1-scores: {}".format(synthetic_f1_scores))
    logging.info("  Synthetic Mean Accuracy: {}".format(np.mean(synthetic_accuracies)))
    logging.info("  Synthetic Mean Precision: {}".format(np.mean(synthetic_precisions)))
    logging.info("  Synthetic Mean Recall: {}".format(np.mean(synthetic_recalls)))
    logging.info("  Synthetic Mean F1 Score: {}".format(np.mean(synthetic_f1_scores)))
    logging.info("  Synthetic Standard Deviation of Accuracy: {}".format(np.std(synthetic_accuracies)))
    logging.info("  Synthetic Standard Deviation of Precision: {}".format(np.std(synthetic_precisions)))
    logging.info("  Synthetic Standard Deviation of Recall: {}".format(np.std(synthetic_recalls)))
    logging.info("  Synthetic Standard Deviation of F1 Score: {}".format(np.std(synthetic_f1_scores)))

    plot_filename = os.path.join(output_dir, f'bars_synthetic_{classifier_type}.pdf')
    create_plot_classifier_metrics(classifier_type, synthetic_accuracies, synthetic_precisions, synthetic_recalls,
                                   synthetic_f1_scores, plot_filename,
                                   title=f'{title_output_label}_SYNTHETIC')

    logging.info(f" Overall Real Results:")

    logging.info("  Real List of Accuracies: {}".format(real_accuracies))
    logging.info("  Real List of Precisions: {}".format(real_precisions))
    logging.info("  Real List of Recalls: {}".format(real_recalls))
    logging.info("  Real List of F1-scores: {}".format(real_f1_scores))
    logging.info("  Real Mean Accuracy: {}".format(np.mean(real_accuracies)))
    logging.info("  Real Mean Precision: {}".format(np.mean(real_precisions)))
    logging.info("  Real Mean Recall: {}".format(np.mean(real_recalls)))
    logging.info("  Real Mean F1 Score: {}".format(np.mean(real_f1_scores)))
    logging.info("  Real Standard Deviation of Accuracy: {}".format(np.std(real_accuracies)))
    logging.info("  Real Standard Deviation of Precision: {}".format(np.std(real_precisions)))
    logging.info("  Real Standard Deviation of Recall: {}".format(np.std(real_recalls)))
    logging.info("  Real Standard Deviation of F1 Score: {}".format(np.std(real_f1_scores)))

    plot_filename = os.path.join(output_dir, f'bars_real_{classifier_type}.pdf')
    create_plot_classifier_metrics(classifier_type, real_accuracies, real_precisions, real_recalls, real_f1_scores, plot_filename,
                                   title=f'{title_output_label}_REAL')


def get_adversarial_model(latent_dim, output_data_shape, activation_function, initializer_mean,
                          initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                          last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d, dataset_type,
                          training_algorithm):
    instance_models = ConditionalGAN(latent_dim, output_data_shape, activation_function, initializer_mean,
                                     initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                     last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d, dataset_type)

    generator_model = instance_models.get_generator()
    discriminator_model = instance_models.get_discriminator()

    adversarial_model = AdversarialModel(generator_model, discriminator_model, latent_dimension=latent_dim)

    optimizer_generator = adversarial_model.get_optimizer(training_algorithm)
    optimizer_discriminator = adversarial_model.get_optimizer(training_algorithm)
    loss_generator = BinaryCrossentropy()
    loss_discriminator = BinaryCrossentropy()

    adversarial_model.compile(optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator)

    return adversarial_model


def run_experiment(dataset, input_data_shape, k, classifier_type, output_dir, batch_size, training_algorithm,
                   number_epochs, latent_dim, activation_function, dropout_decay_rate_g, dropout_decay_rate_d,
                   dense_layer_sizes_g=None, dense_layer_sizes_d=None, dataset_type=None, output_label=None,
                   initializer_mean=None, initializer_deviation=None,
                   last_layer_activation=DEFAULT_CONDITIONAL_LAST_ACTIVATION_LAYER,
                   save_models=False, path_confusion_matrix=None, path_curve_loss=None):

    stratified = StratifiedKFold(n_splits=k, shuffle=True)

    accuracies, precisions, recalls, f1_scores = [], [], [], []
    real_accuracies, real_precisions, real_recalls, real_f1_scores = [], [], [], []

    for i, (train_index, test_index) in enumerate(stratified.split(dataset.iloc[:, :-1], dataset.iloc[:, -1])):

        adversarial_model = get_adversarial_model(latent_dim, input_data_shape, activation_function, initializer_mean,
                                                  initializer_deviation, dropout_decay_rate_g, dropout_decay_rate_d,
                                                  last_layer_activation, dense_layer_sizes_g, dense_layer_sizes_d,
                                                  dataset_type,
                                                  training_algorithm)

        x_training = np.array(dataset.iloc[train_index, :-1].values, dtype=dataset_type)
        x_test = np.array(dataset.iloc[test_index, :-1].values, dtype=dataset_type)

        y_training = np.array(dataset.iloc[train_index, -1].values, dtype=dataset_type)

        y_test = np.array(dataset.iloc[test_index, -1].values, dtype=dataset_type)

        x_training = x_training[0:int(len(x_training) - (len(x_training) % batch_size)), :]
        y_training = y_training[0:int(len(y_training) - (len(y_training) % batch_size))]

        training_history = adversarial_model.fit(x_training, y_training, epochs=number_epochs, batch_size=batch_size)

        if save_models:
            adversarial_model.save_models(output_dir, i)

        plot_training_loss_curve(training_history.history['loss_g'], training_history.history['loss_d'], output_dir,
                                 i, path_curve_loss)

        number_samples_true = len([positional_label for positional_label in y_test.tolist() if positional_label == 1])
        number_samples_false = len([positional_label for positional_label in y_test.tolist() if positional_label == 0])

        x_true_synthetic, y_true_synthetic = generate_samples(adversarial_model, number_samples_true, latent_dim, 1)
        x_false_synthetic, y_false_synthetic = generate_samples(adversarial_model, number_samples_false, latent_dim, 0)

        x_synthetic_samples = np.concatenate((x_true_synthetic, x_false_synthetic), dtype=dataset_type)
        # TODO: Importante para corrigir o problema do formato de entrada
        y_synthetic_samples = np.rint(np.concatenate((y_true_synthetic, y_false_synthetic)))
        y_synthetic_samples = np.squeeze(y_synthetic_samples, axis=1)

        instance_classifier = get_classifier(classifier_type, x_training, y_training, input_data_shape, dataset_type)

        evaluation_results_synthetic_data = evaluate_synthetic_data(instance_classifier, x_synthetic_samples,
                                                                    y_synthetic_samples, i, k, True,
                                                                    output_dir, classifier_type, output_label,
                                                                    path_confusion_matrix)

        evaluation_results_real_data = evaluate_real_data(instance_classifier, x_test, y_test, i, k, True,
                                                          output_dir, classifier_type, output_label,
                                                          path_confusion_matrix)

        accuracies.append(evaluation_results_synthetic_data[0])
        precisions.append(evaluation_results_synthetic_data[1])
        recalls.append(evaluation_results_synthetic_data[2])
        f1_scores.append(evaluation_results_synthetic_data[3])

        real_accuracies.append(evaluation_results_real_data[0])
        real_precisions.append(evaluation_results_real_data[1])
        real_recalls.append(evaluation_results_real_data[2])
        real_f1_scores.append(evaluation_results_real_data[3])

    show_and_export_results(accuracies, precisions, recalls, f1_scores, real_accuracies, real_precisions, real_recalls,
                            real_f1_scores, classifier_type, output_dir, output_label)


def initial_step(initial_arguments, dataset_type):

    Path(initial_arguments.output_dir).mkdir(parents=True, exist_ok=True)
    file_args = os.path.join(initial_arguments.output_dir, 'commandline_args')

    with open(file_args, 'w') as f:
        json.dump(initial_arguments.__dict__, f, indent=2)

    dataset_file_loaded = pd.read_csv(initial_arguments.input_dataset, dtype=dataset_type)
    dataset_file_loaded = dataset_file_loaded.dropna()
    dataset_input_shape = dataset_file_loaded.shape[1] - 1

    input_dataset = os.path.basename(initial_arguments.input_dataset)
    dataset_labels = f'{input_dataset}_{initial_arguments.data_type}_{initial_arguments.classifier}'

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
    parser.add_argument('-c', '--classifier', type=str, required=True,
                        choices=['knn', 'perceptron', 'random_forest', 'svm'],
                        help='Classificador a ser utilizado (knn, perceptron, random_forest, svm).')
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
    parser.add_argument("--training_algorithm", type=str,
                        default=DEFAULT_ADVERSARIAL_TRAINING_ALGORITHM,
                        help="Algoritmo de treinamento para cGAN.",
                        choices=['Adam', 'RMSprop', 'Adadelta'])
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
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Opção para usar a GPU para treinamento.')
    parser.add_argument('--batch_size', type=int,
                        default=DEFAULT_ADVERSARIAL_BATCH_SIZE,
                        choices=[16, 32, 64],
                        help='Tamanho do lote da cGAN.')
    parser.add_argument('--output_format_plot', type=str, default='pdf',
                        choices=['pdf', 'png'],
                        help='Formato de saída para o gráfico (pdf ou png). Default: pdf')
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

    return parser.parse_args()


if __name__ == "__main__":

    arguments = create_argparse()

    if arguments.verbosity == logging.DEBUG:
        logging.basicConfig(format="%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
                            datefmt=TIME_FORMAT, level=arguments.verbosity)
        show_all_settings(arguments)

    else:

        logging.basicConfig(format="%(message)s", datefmt=TIME_FORMAT, level=arguments.verbosity)
        show_all_settings(arguments)

    time_start_campaign = datetime.datetime.now()

    if arguments.data_type == 'int8':
        data_type = np.int8

    elif arguments.data_type == 'float16':
        data_type = np.float16

    else:
        data_type = np.float32

    if args.dense_layer_sizes_g != DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G:
    	args.dense_layer_sizes_g = args.dense_layer_sizes_g[0]

    if args.dense_layer_sizes_d != DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D:
    	args.dense_layer_sizes_d = args.dense_layer_sizes_d[0]

    dataset_file, output_shape, output_label = initial_step(arguments, data_type)

    run_experiment(dataset_file, output_shape,
                   arguments.k_fold,
                   arguments.classifier, arguments.output_dir, batch_size=arguments.batch_size,
                   training_algorithm=arguments.training_algorithm, number_epochs=arguments.number_epochs,
                   latent_dim=arguments.latent_dimension, activation_function=arguments.activation_function,
                   dropout_decay_rate_g=arguments.dropout_decay_rate_g,
                   dropout_decay_rate_d=arguments.dropout_decay_rate_d,
                   dense_layer_sizes_g=arguments.dense_layer_sizes_g, dense_layer_sizes_d=arguments.dense_layer_sizes_d,
                   dataset_type=data_type, output_label=output_label, initializer_mean=arguments.initializer_mean,
                   initializer_deviation=arguments.initializer_deviation, save_models=arguments.save_models,
                   path_confusion_matrix=arguments.path_confusion_matrix, path_curve_loss=arguments.path_curve_loss)

    time_end_campaign = datetime.datetime.now()
    logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))
