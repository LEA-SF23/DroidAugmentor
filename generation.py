import os
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
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
import argparse
import plotly.graph_objects as go
import statistics
import plotly.io as pio
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*the default value of `keepdims` will become False.*")


def main(args):
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Usar a GPU com ID 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Desabilitar o uso de GPU



def create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores):
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
        title='',
        yaxis=dict(title='Média %'),
        xaxis=dict(title='Desempenho do classificador'),
        showlegend=False,
        plot_bgcolor='white'  # Define a cor de fundo para gelo (RGB: 240, 240, 240)
    )

    plot_filename = f'{classifier_type}_classifier.pdf'
    pio.write_image(fig, plot_filename, format='pdf')

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



def run_experiment(df_new, num_samples_class1, num_samples_class0, out_sh, k, classifier_type, training_algorithm='Adam', num_epochs=10000, latent_dim=128, activation_function='LeakyReLU', dropout_decay_rate_g=0.2, dropout_decay_rate_d=0.4, dense_layer_sizes_g=[128, 256, 512], dense_layer_sizes_d=[512, 256, 128]):
    cgan = cGAN(latent_dim=latent_dim, out_shape=out_sh, training_algorithm=training_algorithm, activation_function=activation_function, dropout_decay_rate_g=dropout_decay_rate_g, dropout_decay_rate_d=dropout_decay_rate_d, dense_layer_sizes_g=dense_layer_sizes_g, dense_layer_sizes_d=dense_layer_sizes_d)   
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(df_new.iloc[:, :-1], df_new.iloc[:, -1])):
        X_train, X_test = np.array(df_new.iloc[train_index, :-1].values, dtype=np.int8), np.array(df_new.iloc[test_index, :-1].values, dtype=np.int8)
        y_train, y_test = np.array(df_new.iloc[train_index, -1].values, dtype=np.int8), np.array(df_new.iloc[test_index, -1].values, dtype=np.int8)

        cgan.train(X_train, y_train, num_samples_class1, num_samples_class0, epochs=num_epochs, plot=False)

        df_positive_synthetic = generate_instances(cgan, num_samples_class1, 1)
        df_negative_synthetic = generate_instances(cgan, num_samples_class0, 0)
        df_synthetic = pd.concat([df_positive_synthetic, df_negative_synthetic], ignore_index=True)

        
        synthetic_train_idx = np.random.choice(df_synthetic.index, size=int(0.8 * len(df_synthetic)), replace=False)
        synthetic_test_idx = np.setdiff1d(df_synthetic.index, synthetic_train_idx)

        X_synthetic_train = np.array(df_synthetic.loc[synthetic_train_idx].iloc[:, :-1].values, dtype=np.int8)
        y_synthetic_train = np.array(df_synthetic.loc[synthetic_train_idx].iloc[:, -1].values, dtype=np.int8)
        X_synthetic_test = np.array(df_synthetic.loc[synthetic_test_idx].iloc[:, :-1].values, dtype=np.int8)
        y_synthetic_test = np.array(df_synthetic.loc[synthetic_test_idx].iloc[:, -1].values, dtype=np.int8)

        if classifier_type == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=5)
        elif classifier_type == 'perceptron':
            classifier = perceptron(out_shape)  
            # treinar perceptron com 20 epochs
            classifier.fit(X_train, y_train, epochs=500)  
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_type == 'svm':
            # Hyperparameter tuning for SVM
            parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
            svm_classifier = SVC()
            classifier = GridSearchCV(svm_classifier, parameters, cv=3)
        else:
            raise ValueError("Invalid classifier type. Use 'knn', 'perceptron', 'random_forest', or 'svm'.")

        classifier.fit(X_train, y_train)
        y_pred_synthetic = classifier.predict(X_synthetic_test)

       
        y_synthetic_test = y_synthetic_test.astype(int)
        y_pred_synthetic = y_pred_synthetic.astype(int)

        cm_synthetic = confusion_matrix(y_synthetic_test, y_pred_synthetic)
        accuracy_synthetic = accuracy_score(y_synthetic_test, y_pred_synthetic)  # 
        precision_synthetic = precision_score(y_synthetic_test, y_pred_synthetic)  #
        recall_synthetic = recall_score(y_synthetic_test, y_pred_synthetic)  # 
        f1_synthetic = f1_score(y_synthetic_test, y_pred_synthetic)
	# Obtendo o diretório atual do script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Salvar os dados sintéticos em um arquivo CSV
        synthetic_filename = f'synthetic_data_fold_{i+1}.csv'

        # Verifique se é a última dobra
        if i + 1 == k:
            # Obtendo o diretório atual do script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            synthetic_filepath = os.path.join(script_dir, synthetic_filename)
            df_synthetic.to_csv(synthetic_filepath, index=False, sep=',', header=True)

        
        accuracies.append(accuracy_synthetic)
        precisions.append(precision_synthetic)
        recalls.append(recall_synthetic)
        f1_scores.append(f1_synthetic)

        print(f"Fold {i + 1} - Metrics:")
        print("Confusion Matrix:")
        print(cm_synthetic)
        print("Accuracy:", accuracy_synthetic)
        print("Precision:", precision_synthetic)
        print("Recall:", recall_synthetic)
        print("F1 Score:", f1_synthetic)
        print("---")

    # Depois de todas as dobras, imprima e salve a média e o desvio padrão das métricas
    print("List of Accuracies:", accuracies)
    print("List of Precisions:", precisions)
    print("List of Recalls:", recalls)
    print("List of F1-scores:", f1_scores)
    print("Mean Accuracy:", np.mean(accuracies))
    print("Mean Precision:", np.mean(precisions))
    print("Mean Recall:", np.mean(recalls))
    print("Mean F1 Score:", np.mean(f1_scores))
    print("Standard Deviation of Accuracy:", np.std(accuracies))
    print("Standard Deviation of Precision:", np.std(precisions))
    print("Standard Deviation of Recall:", np.std(recalls))
    print("Standard Deviation of F1 Score:", np.std(f1_scores))

	

    # plotar gráfico com nome do classificador
    create_and_save_plot(classifier_type, accuracies, precisions, recalls, f1_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiment with cGAN and classifiers')
    parser.add_argument('--input_dataset', type=str, required=True, help='Diretório do dataset.')
    parser.add_argument('--data_type', type=str, default='float32', choices=['int8', 'float16', 'float32'], help='Tipo de dado para representar as características das amostras.')
    parser.add_argument('--output_dataset', type=str, required=True, help='Diretório de saída para o dataset sintético gerado.')
    parser.add_argument('--num_samples_class_malware', type=int, default=None, help='Número de amostras da Classe 1 (maligno).')
    parser.add_argument('--num_samples_class_benign', type=int, default=None, help='Número de amostras da Classe 0 (benigno).')
    parser.add_argument('--number_epochs', type=int, default=10000, help='Número de épocas (iterações de treinamento).')
    parser.add_argument('--classifier', type=str, required=True, choices=['knn', 'perceptron', 'random_forest', 'svm'], help='Classificador a ser utilizado (knn, perceptron, random_forest, svm).')
    parser.add_argument('--k_fold', type=int, default=5, help='Número de folds para validação cruzada.')
    parser.add_argument("--latent_dimension", type=int, default=128, help="Dimensão do espaço latente para treinamento cGAN")
    parser.add_argument("--training_algorithm", type=str, default='Adam', choices=['Adam', 'RMSprop', 'Adadelta'], help="Algoritmo de treinamento para cGAN.")
    parser.add_argument("--activation_function", type=str, default='LeakyReLU', choices=['LeakyReLU', 'ReLU', 'PReLU'], help="Função de ativação da cGAN.")
    parser.add_argument("--dropout_decay_rate_g", type=float, default=0.2, help="Taxa de decaimento do dropout do gerador da cGAN")
    parser.add_argument("--dropout_decay_rate_d", type=float, default=0.4, help="Taxa de decaimento do dropout do discriminador da cGAN")  
    parser.add_argument("--dense_layer_sizes_g", type=int, nargs='+', default=[128, 256, 512], help=" Valor das camadas densas do gerador")
    parser.add_argument("--dense_layer_sizes_d", type=int, nargs='+', default=[512, 256, 128], help="valor das camadas densas do discriminador")
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Opção para usar a GPU do TensorFlow.')
	

    args = parser.parse_args()

    # Carregando o conjunto de dados (df_new)
    df_new = pd.read_csv(args.input_dataset, dtype=args.data_type)
    df_new = df_new.dropna()

    out_shape = df_new.shape[1] - 1
    

    # Executando o experimento com k-fold
    run_experiment(df_new, args.num_samples_class_malware, args.num_samples_class_benign, out_shape, args.k_fold, args.classifier, training_algorithm=args.training_algorithm, num_epochs=args.number_epochs, latent_dim=args.latent_dimension, activation_function=args.activation_function, dropout_decay_rate_g=args.dropout_decay_rate_g, dropout_decay_rate_d=args.dropout_decay_rate_d, dense_layer_sizes_g=args.dense_layer_sizes_g, dense_layer_sizes_d=args.dense_layer_sizes_d)
