#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'unknown'
__email__ = 'unknown@unknown.com.br'
__version__ = '{1}.{0}.{0}'
__initial_data__ = '2022/06/01'
__last_update__ = '2023/08/03'
__credits__ = ['unknown']

import os
import statistics
import numpy as np
import itertools

import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

from scipy.special import rel_entr
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise

DEFAULT_MATRIX_CONFUSION_CLASS_LABELS = ["Maligno", "Benigno"]
DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS = ["Rótulo Verdadeiro", "Rótulo Predito"]
DEFAULT_MATRIX_CONFUSION_TITLE = "Matriz de Confusão"
DEFAULT_WIDTH_BAR = 0.2
DEFAULT_FONT_SIZE = 12
DEFAULT_MATRIX_CONFUSION_ROTATION_LEGENDS = 45
DEFAULT_LOSS_CURVE_LEGEND_GENERATOR = "Gerador"
DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR = "Discriminador"
DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS = "Interações (Épocas)"
DEFAULT_LOSS_CURVE_TITLE_PLOT = "Perda do Gerador e Discriminador"
DEFAULT_LOSS_CURVE_LEGEND_LOSS = "Perda"
DEFAULT_LOSS_CURVE_LEGEND_NAME = "Legenda"
DEFAULT_LOSS_CURVE_PREFIX_FILE = "curve_training_error"
DEFAULT_TITLE_COMPARATIVE_PLOTS = "Comparativo entre dados sintéticos e reais (Média)"
DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
DEFAULT_PLOT_REGRESSION_METRICS_LABELS = ['Erro Médio Quadrático', 'Similaridade de Cossenos', 'Divergência KL',
                                          'Máxima Discrepância Média']
DEFAULT_COLOR_MAP = ['#3182BD', '#6BAED6', '#FD8D3C', '#FDD0A2', '#31A354', '#74C476', '#E6550D', '#FD8D3C']
DEFAULT_COLOR_NAME = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
                          'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


class PlotConfusionMatrix:

    def __init__(self, class_labels=None, titles_confusion_matrix_labels=None,
                 title_confusion_matrix=DEFAULT_MATRIX_CONFUSION_TITLE,
                 legend_rotation=DEFAULT_MATRIX_CONFUSION_ROTATION_LEGENDS):

        if titles_confusion_matrix_labels is None:
            titles_confusion_matrix_labels = DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS

        if class_labels is None:
            class_labels = DEFAULT_MATRIX_CONFUSION_CLASS_LABELS

        self.class_labels = class_labels
        self.titles_confusion_matrix = titles_confusion_matrix_labels
        self.title_confusion_matrix = title_confusion_matrix
        self.legend_rotation = legend_rotation

    def plot_confusion_matrix(self, confusion_matrix, confusion_matrix_title=None, cmap=None):

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

        if confusion_matrix_title is None:
            confusion_matrix_title = self.title_confusion_matrix

        plt.title(confusion_matrix_title)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_labels))
        plt.xticks(tick_marks, self.class_labels, rotation=self.legend_rotation)
        plt.yticks(tick_marks, self.class_labels)
        thresh = confusion_matrix.max() / 2.

        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(self.titles_confusion_matrix[0], fontsize=12)
        plt.xlabel(self.titles_confusion_matrix[1], fontsize=12)

    def set_class_labels(self, class_labels):
        self.class_labels = class_labels

    def set_titles_confusion_matrix(self, titles_confusion_matrix):
        self.titles_confusion_matrix = titles_confusion_matrix

    def set_title_confusion_matrix(self, title_confusion_matrix):
        self.title_confusion_matrix = title_confusion_matrix

    def set_legend_rotation(self, legend_rotation):
        self.legend_rotation = legend_rotation


class PlotCurveLoss:

    def __init__(self, loss_curve_legend_generator=DEFAULT_LOSS_CURVE_LEGEND_GENERATOR,
                 loss_curve_legend_discriminator=DEFAULT_LOSS_CURVE_LEGEND_DISCRIMINATOR,
                 loss_curver_title_plot=DEFAULT_LOSS_CURVE_TITLE_PLOT,
                 loss_curve_legend_iterations=DEFAULT_LOSS_CURVE_LEGEND_ITERATIONS,
                 loss_curve_legend_loss=DEFAULT_LOSS_CURVE_LEGEND_LOSS,
                 loss_curve_legend_name=DEFAULT_LOSS_CURVE_LEGEND_NAME,
                 loss_curve_prefix_file=DEFAULT_LOSS_CURVE_PREFIX_FILE):
        self.loss_curve_legend_generator = loss_curve_legend_generator
        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator
        self.loss_curver_title_plot = loss_curver_title_plot
        self.loss_curve_legend_iterations = loss_curve_legend_iterations
        self.loss_curve_legend_loss = loss_curve_legend_loss
        self.loss_curve_legend_name = loss_curve_legend_name
        self.loss_curve_prefix_file = loss_curve_prefix_file

    def plot_training_loss_curve(self, generator_loss, discriminator_loss, output_dir, k_fold, path_curve_loss):
        if output_dir is not None:
            new_loss_curve_plot = go.Figure()
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(generator_loss))), y=generator_loss,
                                                     name=self.loss_curve_legend_generator))
            new_loss_curve_plot.add_trace(go.Scatter(x=list(range(len(discriminator_loss))), y=discriminator_loss,
                                                     name=self.loss_curve_legend_discriminator))

            new_loss_curve_plot.update_layout(title=self.loss_curver_title_plot,
                                              xaxis_title=self.loss_curve_legend_iterations,
                                              yaxis_title=self.loss_curve_legend_loss,
                                              legend_title=self.loss_curve_legend_name)

            Path(os.path.join(output_dir, path_curve_loss)).mkdir(parents=True, exist_ok=True)
            file_name_output = self.loss_curve_prefix_file + "_k_{}.pdf".format(str(k_fold + 1))
            pio.write_image(new_loss_curve_plot, os.path.join(output_dir, path_curve_loss, file_name_output))

    def set_loss_curve_legend_generator(self, loss_curve_legend_generator):
        self.loss_curve_legend_generator = loss_curve_legend_generator

    def set_loss_curve_legend_discriminator(self, loss_curve_legend_discriminator):
        self.loss_curve_legend_discriminator = loss_curve_legend_discriminator

    def set_loss_curver_title_plot(self, loss_curver_title_plot):
        self.loss_curver_title_plot = loss_curver_title_plot

    def set_loss_curve_legend_iterations(self, loss_curve_legend_iterations):
        self.loss_curve_legend_iterations = loss_curve_legend_iterations

    def set_loss_curve_legend_loss(self, loss_curve_legend_loss):
        self.loss_curve_legend_loss = loss_curve_legend_loss

    def set_loss_curve_legend_name(self, loss_curve_legend_name):
        self.loss_curve_legend_name = loss_curve_legend_name

    def set_loss_curve_prefix_file(self, loss_curve_prefix_file):
        self.loss_curve_prefix_file = loss_curve_prefix_file


class PlotClassificationMetrics:

    def __init__(self, labels_bar_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR,
                 font_size=DEFAULT_FONT_SIZE):

        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP

        if labels_bar_metrics is None:
            labels_bar_metrics = DEFAULT_PLOT_CLASSIFIER_METRICS_LABELS

        self.labels_bar_metrics = labels_bar_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.font_size = font_size

    def plot_classifier_metrics(self, classifier_type, accuracy_list, precision_list, recall_list, f1_score_list,
                                plot_filename, plot_title):

        list_all_metrics = [accuracy_list, precision_list, recall_list, f1_score_list]

        new_plot_bars = go.Figure()

        for metric, metric_values, color in zip(self.labels_bar_metrics, list_all_metrics, self.color_map_bar):
            try:
                metric_mean = statistics.mean(metric_values)
                metric_std = statistics.stdev(metric_values)
    
                new_plot_bars.add_trace(go.Bar(x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                                               error_y=dict(type='constant', value=metric_std, visible=True),
                                               width=self.width_bar))
                new_plot_bars.add_annotation(x=metric, y=metric_mean + metric_std, xref="x", yref="y",
                                             text=f' {metric_std:.4f}', showarrow=False,
                                             font=dict(color='black', size=self.font_size),
                                             xanchor='center', yanchor='bottom')
            except Exception as e:
              print("Metric {} error: {}".format(metric, e))

        y_label_dictionary = dict(title=f'Média {len(accuracy_list)} dobras', tickmode='linear', tick0=0.0, dtick=0.1,
                                  gridcolor='black', gridwidth=.05)

        new_plot_bars.update_layout(barmode='group', title=plot_title, yaxis=y_label_dictionary,
                                    xaxis=dict(title=f'Desempenho com {classifier_type}'), showlegend=False,
                                    plot_bgcolor='white')

        pio.write_image(new_plot_bars, plot_filename)

    def set_labels_bar_metrics(self, labels_bar_metrics):
        self.labels_bar_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        self.font_size = font_size


class PlotRegressiveMetrics:

    def __init__(self, labels_plot_regressive_metrics=None, color_map_bar=None, width_bar=DEFAULT_WIDTH_BAR,
                 font_size=DEFAULT_FONT_SIZE, plot_title=DEFAULT_TITLE_COMPARATIVE_PLOTS):

        if color_map_bar is None:
            color_map_bar = DEFAULT_COLOR_MAP

        if labels_plot_regressive_metrics is None:
            labels_plot_regressive_metrics = DEFAULT_PLOT_REGRESSION_METRICS_LABELS

        self.labels_plot_regressive_metrics = labels_plot_regressive_metrics
        self.color_map_bar = color_map_bar
        self.width_bar = width_bar
        self.plot_title_axis_x = plot_title
        self.font_size = font_size

    def plot_regressive_metrics(self, mean_squared_error_list, list_cosine_similarity,
                                list_kl_divergence, list_max_mean_discrepancy, plot_filename, plot_title):

        list_metrics = [mean_squared_error_list, list_cosine_similarity, list_kl_divergence, list_max_mean_discrepancy]

        new_plot_bars = go.Figure()

        for metric, metric_values, color in zip(self.labels_plot_regressive_metrics, list_metrics, self.color_map_bar):
            try:

              print("Metric: {} values: {} color: {}".format(metric, metric_values, color))
               
              metric_mean = statistics.mean(metric_values)
              metric_std = statistics.stdev(metric_values)
  
              new_plot_bars.add_trace(go.Bar(x=[metric], y=[metric_mean], name=metric, marker=dict(color=color),
                                             error_y=dict(type='constant', value=metric_std, visible=True),
                                             width=self.width_bar))
  
              new_plot_bars.add_annotation(x=metric, y=metric_mean + metric_std, xref="x", yref="y",
                                           text=f' {metric_std:.4f}', showarrow=False,
                                           font=dict(color='black', size=self.font_size),
                                           xanchor='center', yanchor='bottom')
            except Exception as e:
              print("Metric: {} Exception: {}".format(metric, e))

        y_label_dictionary = dict(title=f'Média {len(mean_squared_error_list)} dobras', tickmode='linear', tick0=0.0,
                                  dtick=0.1, gridcolor='black', gridwidth=.05)

        new_plot_bars.update_layout(barmode='group', title=plot_title, yaxis=y_label_dictionary,
                                    xaxis=dict(title=self.plot_title_axis_x), showlegend=False,
                                    plot_bgcolor='white')

        pio.write_image(new_plot_bars, plot_filename)

    def set_labels_bar_metrics(self, labels_bar_metrics):
        self.labels_plot_regressive_metrics = labels_bar_metrics

    def set_color_map_bar(self, color_map_bar):
        self.color_map_bar = color_map_bar

    def set_width_bar(self, width_bar):
        self.width_bar = width_bar

    def set_font_size(self, font_size):
        self.font_size = font_size


class ProbabilisticMetrics:

    def __init__(self):
        pass

    @staticmethod
    def get_mean_squared_error(real_label, predicted_label):
        return mean_squared_error(real_label, predicted_label)

    @staticmethod
    def get_cosine_similarity(real_label, predicted_label):
        return sum(pairwise.cosine_similarity(real_label, predicted_label)[0]) / len(real_label)

    @staticmethod
    def get_kl_divergence(real_label, predicted_label):
        return sum(rel_entr(real_label, predicted_label))

    @staticmethod
    def get_maximum_mean_discrepancy(real_label, predicted_label):
        delta = real_label.mean(0) - predicted_label.mean(0)
        return delta.dot(delta.T)

    @staticmethod
    def get_accuracy(real_label, predicted_label):
        return accuracy_score(real_label, predicted_label)

    @staticmethod
    def get_precision(real_label, predicted_label):
        return precision_score(real_label, predicted_label)

    @staticmethod
    def get_recall(real_label, predicted_label):
        return recall_score(real_label, predicted_label)

    @staticmethod
    def get_f1_score(real_label, predicted_label):
        return f1_score(real_label, predicted_label)
