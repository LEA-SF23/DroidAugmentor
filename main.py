#!/usr/bin/python3
# -*- coding: utf-8 -*-

try:

    import json
    import os
    import sys
    import warnings
    import logging
    import contextlib
    import numpy as np
    import pandas as pd
    import datetime
    import itertools

    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.metrics import precision_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    from Models.ConditionalGANModel import ConditionalGAN

    import tensorflow as tf
    from tensorflow import keras

    from keras.layers import MaxPooling2D
    from keras.layers import Conv2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.utils import to_categorical

    import argparse
    import plotly.graph_objects as go
    import statistics
    import plotly.io as pio
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from pathlib import Path
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf_logger = logging.getLogger('tensorflow')
# tf_logger.setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=FutureWarning)
#
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")

DEFAULT_DATA_TYPE = "float32"
DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES = 2000
DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES = 2000
DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN = 10000
DEFAULT_NUMBER_STRATIFICATION_FOLD = 5

DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION = 128
DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM = "Adam"
DEFAULT_CONDITIONAL_GAN_ACTIVATION = "LeakyReLU"
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G = 0.2
DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D = 0.4
DEFAULT_CONDITIONAL_GAN_BATCH_SIZE = 32
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G = [256, 256, 512, 512]
DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D = [512, 512, 256, 256]

DEFAULT_PERCEPTRON_TRAINING_ALGORITHM = "Adam"
DEFAULT_PERCEPTRON_LOSS = "binary_crossentropy"
DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS = [512, 256, 256]
DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE = 0.2
DEFAULT_PERCEPTRON_METRIC = ["accuracy"]

DEFAULT_PLOT_BAR_METRICS_LABELS = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
DEFAULT_PLOT_BAR_COLOR_MAP = ['#3182BD', '#6BAED6', '#FD8D3C', '#FDD0A2', '#31A354', '#74C476', '#E6550D', '#FD8D3C']

DEFAULT_MATRIX_CONFUSION_CLASS_LABELS = ["Maligno", "Benigno"]
DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS = ["Rótulo Verdadeiro", "Rótulo Predito"]
DEFAULT_COLOR_NAME_MAP = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
                          'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

# Define a custom argument type for a list of integers
def list_of_ints(arg):
    #return map(int, arg.split(','))[0]
    return [int(x) for x in arg.split(',')]
    
    
def plot_confusion_matrix(cm, normalize=False, title='Matriz de Confusão', cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(DEFAULT_MATRIX_CONFUSION_CLASS_LABELS))
    plt.xticks(tick_marks, DEFAULT_MATRIX_CONFUSION_CLASS_LABELS, rotation=45)
    plt.yticks(tick_marks, DEFAULT_MATRIX_CONFUSION_CLASS_LABELS)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão normalizada")
    else:
        print('Matriz de Confusão não normalizada')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS[0], fontsize=12)
    plt.xlabel(DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS[1], fontsize=12)


def create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename, title):
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


def perceptron(input_shape, data_type_input=np.float32):
    inputs = keras.layers.Input(shape=input_shape, dtype=data_type_input)

    dense_layer = keras.layers.Dense(DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS[0],
                                     activation=keras.activations.swish)(inputs)

    for i in DEFAULT_PERCEPTRON_DENSE_LAYERS_SETTINGS[1:]:
        dense_layer = keras.layers.Dense(i, activation=keras.activations.swish)(dense_layer)
        dense_layer = keras.layers.Dropout(DEFAULT_PERCEPTRON_DROPOUT_DECAY_RATE)(dense_layer)

    dense_layer = keras.layers.Dense(1, activation=keras.activations.sigmoid)(dense_layer)
    model = keras.Model(inputs=inputs, outputs=[dense_layer])
    model.compile(DEFAULT_PERCEPTRON_TRAINING_ALGORITHM, loss=DEFAULT_PERCEPTRON_LOSS,
                  metrics=DEFAULT_PERCEPTRON_METRIC)

    return model


def generate_instances(instance_model, number_instances, latent_dimension, label_class, dataset_type):
    if label_class == 0:

        label_samples_generated_out = np.zeros(number_instances, dtype=dataset_type)
        label_samples_generated = np.ones(number_instances, dtype=dataset_type)
    else:
        label_samples_generated_out = np.ones(number_instances, dtype=dataset_type)
        label_samples_generated = np.zeros(number_instances, dtype=dataset_type)

    random_latent_noise = np.random.normal(0, 0.10, (number_instances, latent_dimension))
    generated_samples = instance_model.instance_generator.predict([random_latent_noise, label_samples_generated_out])
    generated_samples = np.round(generated_samples)

    return generated_samples, label_samples_generated


def get_instance_trained_classifier(classifier_type, x_samples_training, y_samples_training, output_shape,
                                    dataset_type):
    if classifier_type == 'knn':

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(x_samples_training, y_samples_training)

    elif classifier_type == 'perceptron':

        classifier = perceptron(output_shape, dataset_type)
        classifier.fit(x_samples_training, y_samples_training, epochs=30)

    elif classifier_type == 'random_forest':

        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(x_samples_training, y_samples_training)

    elif classifier_type == 'svm':

        parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        svm_classifier = SVC()
        classifier = GridSearchCV(svm_classifier, parameters, cv=3)
        x_samples_training = np.array(x_samples_training, dtype=dataset_type)
        y_samples_training = np.array(y_samples_training, dtype=dataset_type)
        classifier.fit(x_samples_training, y_samples_training)

    else:
        raise ValueError("Invalid classifier type. Use 'knn', 'perceptron', 'random_forest' or 'svm'.")

    return classifier


def evaluate_synthetic_data(classifier, x_synthetic_samples, y_synthetic_samples, fold, k, generate_confusion_matrix,
                            output_dir, classifier_type):
    y_predicted_synthetic = classifier.predict(x_synthetic_samples)

    y_synthetic_samples = y_synthetic_samples.astype(int)
    y_predicted_synthetic = y_predicted_synthetic.astype(int)

    cm_synthetic = confusion_matrix(y_synthetic_samples, y_predicted_synthetic)
    accuracy_synthetic = accuracy_score(y_synthetic_samples, y_predicted_synthetic)
    precision_synthetic = precision_score(y_synthetic_samples, y_predicted_synthetic)
    recall_synthetic = recall_score(y_synthetic_samples, y_predicted_synthetic)
    f1_synthetic = f1_score(y_synthetic_samples, y_predicted_synthetic)

    print(f"Synthetic Fold {fold + 1}/{k} results")
    print(f"Synthetic Fold {fold + 1} - Confusion Matrix:")
    print(cm_synthetic)
    print(f"Synthetic Fold {fold + 1} - Accuracy:", accuracy_synthetic)
    print(f"Synthetic Fold {fold + 1} - Precision:", precision_synthetic)
    print(f"Synthetic Fold {fold + 1} - Recall:", recall_synthetic)
    print(f"Synthetic Fold {fold + 1} - F1 Score:", f1_synthetic)
    print("---")
    print()

    if generate_confusion_matrix:
        plt.figure()
        selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME_MAP[(fold + 2) % len(DEFAULT_COLOR_NAME_MAP)])
        plot_confusion_matrix(cm_synthetic, title=out_label, cmap=selected_color_map)
        matrix_file = os.path.join(output_dir, f'cm_synthetic_{classifier_type}_k{fold + 1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')

    return [accuracy_synthetic, precision_synthetic, recall_synthetic, f1_synthetic]


def evaluate_real_data(classifier, x_real_samples, y_real_samples, fold, k, generate_confusion_matrix, output_dir,
                       classifier_type):
    y_predicted_real = classifier.predict(x_real_samples)

    y_predicted_real = y_predicted_real.astype(int)
    y_sample_real = y_real_samples.astype(int)

    cm_real = confusion_matrix(y_sample_real, y_predicted_real)

    accuracy_real = accuracy_score(y_sample_real, y_predicted_real)
    precision_real = precision_score(y_sample_real, y_predicted_real)
    recall_real = recall_score(y_sample_real, y_predicted_real)
    f1_real = f1_score(y_sample_real, y_predicted_real)

    print(f"Real Fold {fold + 1}/{k} results")
    print(f"Real Fold {fold + 1} results")
    print(f"Real Fold {fold + 1} - Confusion Matrix:")
    print(cm_real)
    print(f"Real Fold {fold + 1} - Accuracy:", accuracy_real)
    print(f"Real Fold {fold + 1} - Precision:", precision_real)
    print(f"Real Fold {fold + 1} - Recall:", recall_real)
    print(f"Real Fold {fold + 1} - F1 Score:", f1_real)
    print("---")
    print()

    if generate_confusion_matrix:
        plt.figure()
        selected_color_map = plt.colormaps.get_cmap(DEFAULT_COLOR_NAME_MAP[(fold + 2) % len(DEFAULT_COLOR_NAME_MAP)])
        plot_confusion_matrix(cm_real, title=out_label, cmap=selected_color_map)
        matrix_file = os.path.join(output_dir, f'cm_real_{classifier_type}_k{fold + 1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')

    return [accuracy_real, precision_real, recall_real, f1_real]


def run_experiment(dataset, number_samples_generate_true_class, number_samples_generate_false_class, output_shape, k,
                   classifier_type, output_dir, batch_size, training_algorithm, number_epochs, latent_dim,
                   activation_function, dropout_decay_rate_g, dropout_decay_rate_d, dense_layer_sizes_g=None,
                   dense_layer_sizes_d=None, dataset_type=None, output_label=None):

    stratified = StratifiedKFold(n_splits=k, shuffle=True)
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    real_accuracies, real_precisions, real_recalls, real_f1_scores = [], [], [], []

    for i, (train_index, test_index) in enumerate(stratified.split(dataset.iloc[:, :-1], dataset.iloc[:, -1])):
        instance_conditional_gan = ConditionalGAN(latent_dim=latent_dim, output_shape=output_shape,
                                                  training_algorithm=training_algorithm,
                                                  activation_function=activation_function,
                                                  dropout_decay_rate_g=dropout_decay_rate_g,
                                                  dropout_decay_rate_d=dropout_decay_rate_d,
                                                  dense_layer_sizes_g=dense_layer_sizes_g,
                                                  dense_layer_sizes_d=dense_layer_sizes_d, batch_size=batch_size,
                                                  dataset_type=dataset_type, 
                                                  output_dir=output_dir,
                                                  output_file=f'curve_trainning_error_{i+1}.pdf'
                                                  )

        x_train = np.array(dataset.iloc[train_index, :-1].values, dtype=dataset_type)
        x_test = np.array(dataset.iloc[test_index, :-1].values, dtype=dataset_type)

        y_train = np.array(dataset.iloc[train_index, -1].values, dtype=dataset_type)
        y_train = y_train.reshape((len(y_train), 1))
        y_test = np.array(dataset.iloc[test_index, -1].values, dtype=dataset_type)

        instance_conditional_gan.train(x_train, y_train, epochs=number_epochs)

        x_df_positive_synthetic, y_df_positive_synthetic = generate_instances(instance_conditional_gan,
                                                                              number_samples_generate_true_class,
                                                                              latent_dim, 1, dataset_type)
        x_df_negative_synthetic, y_df_negative_synthetic = generate_instances(instance_conditional_gan,
                                                                              number_samples_generate_false_class,
                                                                              latent_dim, 0, dataset_type)

        x_synthetic_samples = np.concatenate([x_df_positive_synthetic, x_df_negative_synthetic], dtype=dataset_type)
        y_synthetic_samples = np.concatenate([y_df_positive_synthetic, y_df_negative_synthetic], dtype=dataset_type)

        instance_classifier = get_instance_trained_classifier(classifier_type, x_test, y_test, output_shape,
                                                              dataset_type)

        evaluation_results_synthetic_data = evaluate_synthetic_data(instance_classifier, x_synthetic_samples,
                                                                    y_synthetic_samples, i, k, True,
                                                                    output_dir, classifier_type)
        evaluation_results_real_data = evaluate_real_data(instance_classifier, x_test, y_test, i, k, True,
                                                          output_dir, classifier_type)

        accuracies.append(evaluation_results_synthetic_data[0])
        precisions.append(evaluation_results_synthetic_data[1])
        recalls.append(evaluation_results_synthetic_data[2])
        f1_scores.append(evaluation_results_synthetic_data[3])

        real_accuracies.append(evaluation_results_real_data[0])
        real_precisions.append(evaluation_results_real_data[1])
        real_recalls.append(evaluation_results_real_data[2])
        real_f1_scores.append(evaluation_results_real_data[3])

    print(f"Overall Synthetic Results:")
    print("Synthetic List of Accuracies:", accuracies)
    print("Synthetic List of Precisions:", precisions)
    print("Synthetic List of Recalls:", recalls)
    print("Synthetic List of F1-scores:", f1_scores)
    print("Synthetic Mean Accuracy:", np.mean(accuracies))
    print("Synthetic Mean Precision:", np.mean(precisions))
    print("Synthetic Mean Recall:", np.mean(recalls))
    print("Synthetic Mean F1 Score:", np.mean(f1_scores))
    print("Synthetic Standard Deviation of Accuracy:", np.std(accuracies))
    print("Synthetic Standard Deviation of Precision:", np.std(precisions))
    print("Synthetic Standard Deviation of Recall:", np.std(recalls))
    print("Synthetic Standard Deviation of F1 Score:", np.std(f1_scores))

    plot_filename = os.path.join(output_dir, f'bars_synthetic_{classifier_type}.pdf')
    create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename,
                         title=f'{output_label}_SYNTHETIC')

    print(f"Overall Real Results:")
    print("Real List of Accuracies:", real_accuracies)
    print("Real List of Precisions:", real_precisions)
    print("Real List of Recalls:", real_recalls)
    print("Real List of F1-scores:", real_f1_scores)
    print("Real Mean Accuracy:", np.mean(real_accuracies))
    print("Real Mean Precision:", np.mean(real_precisions))
    print("Real Mean Recall:", np.mean(real_recalls))
    print("Real Mean F1 Score:", np.mean(real_f1_scores))
    print("Real Standard Deviation of Accuracy:", np.std(real_accuracies))
    print("Real Standard Deviation of Precision:", np.std(real_precisions))
    print("Real Standard Deviation of Recall:", np.std(real_recalls))
    print("Real Standard Deviation of F1 Score:", np.std(real_f1_scores))

    plot_filename = os.path.join(output_dir, f'bars_real_{classifier_type}.pdf')
    create_and_save_plot(classifier_type, real_accuracies, real_precisions, real_recalls, real_f1_scores, plot_filename,
                         title=f'{output_label}_REAL')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the experiment with cGAN and classifiers')
    parser.add_argument('-i', '--input_dataset', type=str, required=True, help='Arquivo do dataset de entrada')
    parser.add_argument('-c', '--classifier', type=str, required=True,
                        choices=['knn', 'perceptron', 'random_forest', 'svm'],
                        help='Classificador a ser utilizado (knn, perceptron, random_forest, svm).')
    parser.add_argument('-o', '--output_dir', type=str,
                        default=f'out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Diretório para gravação dos arquivos de saída.')
    parser.add_argument('--data_type', type=str, default=DEFAULT_DATA_TYPE, choices=['int8', 'float16', 'float32'],
                        help='Tipo de dado para representar as características das amostras.')
    parser.add_argument('--num_samples_class_malware', type=int, default=DEFAULT_NUMBER_GENERATE_MALWARE_SAMPLES,
                        help='Número de amostras da Classe 1 (maligno).')
    parser.add_argument('--num_samples_class_benign', type=int, default=DEFAULT_NUMBER_GENERATE_BENIGN_SAMPLES,
                        help='Número de amostras da Classe 0 (benigno).')
    parser.add_argument('--number_epochs', type=int, default=DEFAULT_NUMBER_EPOCHS_CONDITIONAL_GAN,
                        help='Número de épocas (iterações de treinamento).')
    parser.add_argument('--k_fold', type=int, default=DEFAULT_NUMBER_STRATIFICATION_FOLD,
                        help='Número de folds para validação cruzada.')
    parser.add_argument("--latent_dimension", type=int, default=DEFAULT_CONDITIONAL_GAN_LATENT_DIMENSION,
                        help="Dimensão do espaço latente para treinamento cGAN")
    parser.add_argument("--training_algorithm", type=str, default=DEFAULT_CONDITIONAL_GAN_TRAINING_ALGORITHM,
                        help="Algoritmo de treinamento para cGAN.", choices=['Adam', 'RMSprop', 'Adadelta'])
    parser.add_argument("--activation_function", type=str, default=DEFAULT_CONDITIONAL_GAN_ACTIVATION,
                        help="Função de ativação da cGAN.", choices=['LeakyReLU', 'ReLU', 'PReLU'])
    parser.add_argument("--dropout_decay_rate_g", type=float, default=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_G,
                        help="Taxa de decaimento do dropout do gerador da cGAN")
    parser.add_argument("--dropout_decay_rate_d", type=float, default=DEFAULT_CONDITIONAL_GAN_DROPOUT_DECAY_RATE_D,
                        help="Taxa de decaimento do dropout do discriminador da cGAN")
    parser.add_argument("--dense_layer_sizes_g", type=list_of_ints, nargs='+',
                        default=DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G,
                        help=" Valor das camadas densas do gerador")
    parser.add_argument("--dense_layer_sizes_d", type=list_of_ints, nargs='+',
                        default=DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D,
                        help="valor das camadas densas do discriminador")
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Opção para usar a GPU para treinamento.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONDITIONAL_GAN_BATCH_SIZE, choices=[16, 32, 64],
                        help='Tamanho do lote da cGAN.')
    parser.add_argument('--output_format_plot', type=str, default='pdf', choices=['pdf', 'png'],
                        help='Formato de saída para o gráfico (pdf ou png). Default: pdf')

    time_start_campaign = datetime.datetime.now()

    args = parser.parse_args()
    
    
    if args.dense_layer_sizes_g != DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_G:
    	args.dense_layer_sizes_g = args.dense_layer_sizes_g[0]
    
    if args.dense_layer_sizes_d != DEFAULT_CONDITIONAL_GAN_DENSE_LAYERS_SETTINGS_D:
    	args.dense_layer_sizes_d = args.dense_layer_sizes_d[0]
    	
   

    if args.data_type == 'int8':
        data_type = np.int8

    elif args.data_type == 'float16':
        data_type = np.float16

    else:
        data_type = np.float32
    

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    file_args = os.path.join(args.output_dir, 'commandline_args')
    with open(file_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    df_new = pd.read_csv(args.input_dataset, dtype=data_type)
    df_new = df_new.dropna()

    out_shape = df_new.shape[1] - 1

    input_dataset = os.path.basename(args.input_dataset)
    out_label = f'{input_dataset}_{args.data_type}_{args.classifier}'

    run_experiment(df_new, args.num_samples_class_malware, args.num_samples_class_benign, out_shape, args.k_fold,
                   args.classifier, args.output_dir, batch_size=args.batch_size,
                   training_algorithm=args.training_algorithm, number_epochs=args.number_epochs,
                   latent_dim=args.latent_dimension, activation_function=args.activation_function,
                   dropout_decay_rate_g=args.dropout_decay_rate_g, dropout_decay_rate_d=args.dropout_decay_rate_d,
                   dense_layer_sizes_g=args.dense_layer_sizes_g, dense_layer_sizes_d=args.dense_layer_sizes_d,
                   dataset_type=data_type, output_label=out_label)

    time_end_campaign = datetime.datetime.now()
    logging.info("\t Campaign duration: {}".format(time_end_campaign - time_start_campaign))
