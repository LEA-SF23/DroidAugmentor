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
    from sklearn.metrics import precision_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    from cgann import cGAN
    import tensorflow as tf
    from keras.layers import MaxPooling2D, Conv2D
    from keras.layers import Dense, Dropout, Flatten
    from tensorflow import keras
    from keras.utils import to_categorical
    import argparse
    import plotly.graph_objects as go
    import statistics
    import plotly.io as pio
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    from pathlib import Path
    import datetime

    import itertools
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

except ImportError as error:

    print(error)
    print("pip install pipenv; pipenv install -r requirements2.txt")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")


def plot_confusion_matrix(cm,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=None):
    classes = ["Maligno", "Benigno"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão normalizada")
    else:
        print('Matriz de Confusão não normalizada')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo Verdadeiro',  fontsize=12)
    plt.xlabel('Rótulo Predito'   ,  fontsize=12)




def create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename, title):
    metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    values = [accuracies, precisions, recalls, f1_scores]
    colors = ['#3182BD', '#6BAED6', '#FD8D3C', '#FDD0A2', '#31A354', '#74C476', '#E6550D', '#FD8D3C']

    fig = go.Figure()


    for metric, metric_values, color in zip(metrics, values, colors):
        metric_mean = statistics.mean(metric_values)
        metric_std = statistics.stdev(metric_values)

        fig.add_trace(go.Bar(
            x=[metric],
            y=[metric_mean],
            name=metric,
            marker=dict(color=color),
            error_y=dict(type='constant', value=metric_std, visible=True),
            width=0.2,  # Ajuste a largura das barras aqui (valor entre 0 e 1)
        ))

        fig.add_annotation(
            x=metric,
            y=metric_mean + metric_std,
            xref="x",
            yref="y",
            text=f' {metric_std:.4f}',
            showarrow=False,
            font=dict(color='black', size=12),
            xanchor='center',
            yanchor='bottom',
        )

    fig.update_layout(
        barmode='group',
        title=title,
        yaxis=dict(title=f'Média {len(accuracies)} dobras'),
        xaxis=dict(title=f'Desempenho com {classifier_type}'),
        showlegend=False,
        plot_bgcolor='white'  # Define a cor de fundo para gelo (RGB: 240, 240, 240)
    )


    pio.write_image(fig, plot_filename)

def perceptron(out_shape):
    inputs = keras.layers.Input(shape=(out_shape))
    dense_layer = keras.layers.Dense(512, activation=keras.activations.swish)(inputs)
    dense_layer = keras.layers.Dropout(0.2)(dense_layer)
    dense_layer = keras.layers.Dense(64, activation=keras.activations.swish)(dense_layer)
    dense_layer = keras.layers.Dropout(0.2)(dense_layer)
    dense_layer = keras.layers.Dense(64, activation=keras.activations.swish)(dense_layer)
    dense_layer = keras.layers.Dropout(0.2)(dense_layer)
    dense_layer = keras.layers.Dense(64, activation=keras.activations.swish)(dense_layer)
    dense_layer = keras.layers.Dense(1, activation="sigmoid")(dense_layer)  
    model = keras.Model(inputs=inputs, outputs=[dense_layer])
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])  
    return model

def generate_instances(cgan, num_instances, label_class):
    if label_class == 0:
        sampled_labels = np.zeros(num_instances, dtype=np.int8)
    else:
        sampled_labels = np.ones(num_instances, dtype=np.int8)

    noise = np.random.normal(0, 1, (num_instances, 128))
    gen_samples = cgan.generator.predict([noise, sampled_labels])
    gen_samples = np.round(gen_samples)

    gen_df = pd.DataFrame(data=gen_samples, columns=df_new.drop('class', 1).columns)
    gen_df['class'] = sampled_labels  # Use the synthetic labels directly
    return gen_df

def run_experiment(df_new, num_samples_class1, num_samples_class0, out_sh, k, classifier_type, output_format_plot,
                   output_dir, batch_size=32, training_algorithm='Adam', num_epochs=10000, latent_dim=128,
                   activation_function='LeakyReLU', dropout_decay_rate_g=0.2, dropout_decay_rate_d=0.4,
                   dense_layer_sizes_g=[128, 256, 512], dense_layer_sizes_d=[512, 256, 128],
                   arg_dtype=np.int8, out_label=""):

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    real_accuracies = []
    real_precisions = []
    real_recalls = []
    real_f1_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(df_new.iloc[:, :-1], df_new.iloc[:, -1])):
        cgan = cGAN(latent_dim=latent_dim, out_shape=out_sh, training_algorithm=training_algorithm,
                    activation_function=activation_function, dropout_decay_rate_g=dropout_decay_rate_g,
                    dropout_decay_rate_d=dropout_decay_rate_d, dense_layer_sizes_g=dense_layer_sizes_g,
                    dense_layer_sizes_d=dense_layer_sizes_d, batch_size=batch_size)

        X_train, X_test = np.array(df_new.iloc[train_index, :-1].values, dtype=arg_dtype), np.array(df_new.iloc[test_index, :-1].values, dtype=arg_dtype)
        y_train, y_test = np.array(df_new.iloc[train_index, -1].values, dtype=arg_dtype), np.array(df_new.iloc[test_index, -1].values, dtype=arg_dtype)

        cgan.train(X_train, y_train, num_samples_class1, num_samples_class0, epochs=num_epochs, plot=False)

        df_positive_synthetic = generate_instances(cgan, num_samples_class1, 1)
        df_negative_synthetic = generate_instances(cgan, num_samples_class0, 0)
        df_synthetic = pd.concat([df_positive_synthetic, df_negative_synthetic], ignore_index=True)

        synthetic_train_idx = np.random.choice(df_synthetic.index, size=int(0.8 * len(df_synthetic)), replace=False)
        synthetic_test_idx = np.setdiff1d(df_synthetic.index, synthetic_train_idx)

        X_synthetic_train = np.array(df_synthetic.loc[synthetic_train_idx].iloc[:, :-1].values, dtype=arg_dtype)
        y_synthetic_train = np.array(df_synthetic.loc[synthetic_train_idx].iloc[:, -1].values, dtype=arg_dtype)
        X_synthetic_test = np.array(df_synthetic.loc[synthetic_test_idx].iloc[:, :-1].values, dtype=arg_dtype)
        y_synthetic_test = np.array(df_synthetic.loc[synthetic_test_idx].iloc[:, -1].values, dtype=arg_dtype)

        if classifier_type == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=5)
            classifier.fit(X_train, y_train)
        elif classifier_type == 'perceptron':
            classifier = perceptron(out_sh)  
            # treinar perceptron com 20 epochs
            classifier.fit(X_train, y_train, epochs=20)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(X_train, y_train)
        elif classifier_type == 'svm':
            # Hyperparameter tuning for SVM
            parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            svm_classifier = SVC()
            classifier = GridSearchCV(svm_classifier, parameters, cv=3)
            classifier.fit(X_train, y_train)
        else:
            raise ValueError("Invalid classifier type. Use 'knn', 'perceptron', 'random_forest', or 'svm'.")


        #Sintético
        y_pred_synthetic = classifier.predict(X_synthetic_test)

        y_synthetic_test = y_synthetic_test.astype(int)
        y_pred_synthetic = y_pred_synthetic.astype(int)

        cm_synthetic = confusion_matrix(y_synthetic_test, y_pred_synthetic)
        accuracy_synthetic = accuracy_score(y_synthetic_test, y_pred_synthetic)  
        precision_synthetic = precision_score(y_synthetic_test, y_pred_synthetic)  
        recall_synthetic = recall_score(y_synthetic_test, y_pred_synthetic)  
        f1_synthetic = f1_score(y_synthetic_test, y_pred_synthetic)

        synthetic_filepath = os.path.join(output_dir, f'synthetic_dataset_k{i+1}.csv')
        df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)

        accuracies.append(accuracy_synthetic)
        precisions.append(precision_synthetic)
        recalls.append(recall_synthetic)
        f1_scores.append(f1_synthetic)

        print(f"Synthetic Fold {i + 1}/{k} results")
        print(f"Synthetic Fold {i + 1} - Confusion Matrix:")
        print(cm_synthetic)
        print(f"Synthetic Fold {i + 1} - Accuracy:", accuracy_synthetic)
        print(f"Synthetic Fold {i + 1} - Precision:", precision_synthetic)
        print(f"Synthetic Fold {i + 1} - Recall:", recall_synthetic)
        print(f"Synthetic Fold {i + 1} - F1 Score:", f1_synthetic)
        print("---")
        print()

        #cnf_matrix = np.array([[1018, 254], [238, 955]])
        # Plot non-normalized confusion matrix
        plt.figure()
        cmap_names = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        plot_confusion_matrix(cm_synthetic, title=out_label,
                              cmap=plt.colormaps.get_cmap(cmap_names[(i+2) % len(cmap_names)]))
        matrix_file = os.path.join(output_dir, f'cm_synthetic_{classifier_type}_k{i+1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')






        # Real
        y_pred_real = classifier.predict(X_test)

        y_real_test = y_test.astype(int)
        y_pred_real = y_pred_real.astype(int)

        cm_real = confusion_matrix(y_real_test, y_pred_real)
        accuracy_real = accuracy_score(y_real_test, y_pred_real)
        precision_real = precision_score(y_real_test, y_pred_real)
        recall_real = recall_score(y_real_test, y_pred_real)
        f1_real = f1_score(y_real_test, y_pred_real)

        real_accuracies.append(accuracy_real)
        real_precisions.append(precision_real)
        real_recalls.append(recall_real)
        real_f1_scores.append(f1_real)

        print(f"Real Fold {i + 1}/{k} results")
        print(f"Real Fold {i + 1} results")
        print(f"Real Fold {i + 1} - Confusion Matrix:")
        print(cm_synthetic)
        print(f"Real Fold {i + 1} - Accuracy:", accuracy_real)
        print(f"Real Fold {i + 1} - Precision:", precision_real)
        print(f"Real Fold {i + 1} - Recall:", recall_real)
        print(f"Real Fold {i + 1} - F1 Score:", f1_real)
        print("---")
        print()

        # cnf_matrix = np.array([[1018, 254], [238, 955]])
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm_real, title=out_label,
                              cmap=plt.colormaps.get_cmap(cmap_names[(i + 2+k) % len(cmap_names)]))
        matrix_file = os.path.join(output_dir, f'cm_real_{classifier_type}_k{i + 1}.pdf')
        plt.savefig(matrix_file, bbox_inches='tight')



    # Depois de todas as dobras, imprima e salve a média e o desvio padrão das métricas

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
    # plotar gráfico com nome do classificador
    plot_filename = os.path.join(output_dir, f'bars_synthetic_{classifier_type}.pdf')
    create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename,
                         title=f'{out_label}_SYNTHETIC')

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

    # plotar gráfico com nome do classificador
    plot_filename = os.path.join(output_dir, f'bars_real_{classifier_type}.pdf')
    create_and_save_plot(classifier_type, real_accuracies, real_precisions, real_recalls, real_f1_scores, plot_filename,
                         title=f'{out_label}_REAL')

    #TODO gerar arquivos? na verdade, deve ser um processo posterior, com base na rede treinada
    # files_in = os.path.join(output_dir, f'synthetic_*')
    # file_out = os.path.join(output_dir, f'synthetic_dataset_full')
    # os.system(f"cat {files_in} > {file_out}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiment with cGAN and classifiers')
    parser.add_argument('-i', '--input_dataset',  type=str, required=True, help='Arquivo do dataset de entrada')
    parser.add_argument('-c', '--classifier', type=str, required=True, choices=['knn', 'perceptron', 'random_forest', 'svm'],
                        help='Classificador a ser utilizado (knn, perceptron, random_forest, svm).')

    parser.add_argument('-o', '--output_dir', type=str, default=f'out_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                        help='Diretório para gravação dos arquivos de saída.')

    #TODO DONE parece que esse argumento não está sendo utilizado consistentemente
    parser.add_argument('--data_type', type=str, default='float32', choices=['int8', 'float16', 'float32'], help='Tipo de dado para representar as características das amostras.')
    parser.add_argument('--num_samples_class_malware', type=int, default=None, help='Número de amostras da Classe 1 (maligno).')
    parser.add_argument('--num_samples_class_benign', type=int, default=None, help='Número de amostras da Classe 0 (benigno).')
    parser.add_argument('--number_epochs', type=int, default=10000, help='Número de épocas (iterações de treinamento).')
    parser.add_argument('--k_fold', type=int, default=5, help='Número de folds para validação cruzada.')
    parser.add_argument("--latent_dimension", type=int, default=128, help="Dimensão do espaço latente para treinamento cGAN")
    parser.add_argument("--training_algorithm", type=str, default='Adam', choices=['Adam', 'RMSprop', 'Adadelta'], help="Algoritmo de treinamento para cGAN.")
    parser.add_argument("--activation_function", type=str, default='LeakyReLU', choices=['LeakyReLU', 'ReLU', 'PReLU'], help="Função de ativação da cGAN.")
    parser.add_argument("--dropout_decay_rate_g", type=float, default=0.2, help="Taxa de decaimento do dropout do gerador da cGAN")
    parser.add_argument("--dropout_decay_rate_d", type=float, default=0.4, help="Taxa de decaimento do dropout do discriminador da cGAN")  
    parser.add_argument("--dense_layer_sizes_g", type=int, nargs='+', default=[128, 256, 512], help=" Valor das camadas densas do gerador")
    parser.add_argument("--dense_layer_sizes_d", type=int, nargs='+', default=[512, 256, 128], help="valor das camadas densas do discriminador")
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Opção para usar a GPU do TensorFlow.')
    parser.add_argument('--batch_size', type=int, default=32, choices=[16, 32, 64], help='Tamanho do lote da cGAN.')
    parser.add_argument('--output_format_plot', type=str, default='pdf', choices=['pdf', 'png'], help='Formato de saída para o gráfico (pdf ou png). Default: pdf')

    args = parser.parse_args()

    if args.data_type == 'int8':
        arg_dtype = np.int8
    elif args.data_type == 'float16':
        arg_dtype = np.float16
    elif args.data_type == 'float32':
        arg_dtype = np.float32
    else:
        sys.exit()

    # creating a new directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    file_args = os.path.join(args.output_dir, 'commandline_args')
    with open(file_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Carregando o conjunto de dados (df_new)
    df_new = pd.read_csv(args.input_dataset, dtype=arg_dtype)
    df_new = df_new.dropna()

    out_shape = df_new.shape[1] - 1

    input_dataset = os.path.basename(args.input_dataset)
    out_label = f'{input_dataset}_{args.data_type}_{args.classifier}'
    # Executando o experimento com k-fold
    run_experiment(df_new, args.num_samples_class_malware, args.num_samples_class_benign, out_shape, args.k_fold,
                   args.classifier, args.output_format_plot,  args.output_dir, batch_size=args.batch_size,
                   training_algorithm=args.training_algorithm, num_epochs=args.number_epochs,
                   latent_dim=args.latent_dimension, activation_function=args.activation_function,
                   dropout_decay_rate_g=args.dropout_decay_rate_g, dropout_decay_rate_d=args.dropout_decay_rate_d,
                   dense_layer_sizes_g=args.dense_layer_sizes_g, dense_layer_sizes_d=args.dense_layer_sizes_d,
                   arg_dtype=arg_dtype, out_label=out_label)

