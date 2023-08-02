import numpy as np
import itertools
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import statistics

DEFAULT_MATRIX_CONFUSION_CLASS_LABELS = ["Maligno", "Benigno"]
DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS = ["Rótulo Verdadeiro", "Rótulo Predito"]
DEFAULT_MATRIX_CONFUSION_TITLE = "Matriz de Confusão"
DEFAULT_PLOT_BAR_COLOR_MAP = ['#3182BD', '#6BAED6', '#FD8D3C', '#FDD0A2', '#31A354', '#74C476', '#E6550D', '#FD8D3C']
DEFAULT_PLOT_BAR_METRICS_LABELS = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
DEFAULT_COLOR_NAME_MAP = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
                          'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']


class PlotConfusionMatrix:

    def __init__(self, class_labels=None, titles_confusion_matrix_labels=None,
                 title_confusion_matrix=DEFAULT_MATRIX_CONFUSION_TITLE):

        if titles_confusion_matrix_labels is None:
            titles_confusion_matrix_labels = DEFAULT_MATRIX_CONFUSION_PREDICT_LABELS

        self.titles_confusion_labels = titles_confusion_matrix_labels

        if class_labels is None:
            class_labels = DEFAULT_MATRIX_CONFUSION_CLASS_LABELS

        self.class_labels = class_labels
        self.title_confusion_matrix = title_confusion_matrix

    def plot_confusion_matrix(self, confusion_matrix, confusion_matrix_title=None, cmap=None):

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)

        if confusion_matrix_title is None:
            confusion_matrix_title = self.title_confusion_matrix

        plt.title(confusion_matrix_title)
        plt.colorbar()
        tick_marks = np.arange(len(self.class_labels))
        plt.xticks(tick_marks, self.class_labels, rotation=45)
        plt.yticks(tick_marks, self.class_labels)
        thresh = confusion_matrix.max() / 2.

        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel(self.titles_confusion_labels[0], fontsize=12)
        plt.xlabel(self.titles_confusion_labels[1], fontsize=12)


class PlotCurveLoss:

    def __init__(self):
        pass


class PlotClassificationMetrics:

    def __init__(self, labels_bar_metrics=None, color_map_bar=None):

        if color_map_bar is None:
            color_map_bar = DEFAULT_PLOT_BAR_COLOR_MAP

        if labels_bar_metrics is None:
            labels_bar_metrics = DEFAULT_PLOT_BAR_METRICS_LABELS

        self.labels_bar_metrics = labels_bar_metrics
        self.color_map_bar = color_map_bar
    def plot_classifier_metrics(self, classifier_type, accuracies, precisions, recalls, f1_scores, plot_filename,
                                title):
        values = [accuracies, precisions, recalls, f1_scores]

        fig = go.Figure()

        for metric, metric_values, color in zip(self.labels_bar_metrics, values, self.color_map_bar):
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


class PlotDistanceMetrics:

    def __init__(self):
        pass
