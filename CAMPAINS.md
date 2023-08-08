Aqui é apresentado sumariamente os resultados de um experimento realizado em 2023-08-05. 

## Configuração do Experimento para camadas densas de 256. 

1. Comando utilizado para a execução do experimento:
   ```
   main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam

   ```

### Configurações utilizadas:


activation_function       : LeakyReLU
batch_size                : 32
classifier                : ['RandomForest', 'SupportVectorMachine', 'KNN', 'DecisionTree', 'AdaBoost']
data_type                 : float32
dense_layer_sizes_d       : [[256]]
dense_layer_sizes_g       : [[256]]
dropout_decay_rate_d      : 0.4
dropout_decay_rate_g      : 0.2
initializer_deviation     : 0.02
initializer_mean          : 0.0
input_dataset             : datasets/drebin215_original_5560Malwares_6566Benign.csv
k_fold                    : 5
latent_dimension          : 128
latent_mean_distribution  : 0.0
latent_stander_deviation  : 1.0
num_samples_class_benign  : 2000
num_samples_class_malware : 2000
number_epochs             : 1000
output_dir                : outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2
path_confusion_matrix     : confusion_matrix
path_curve_loss           : training_curve
save_models               : True
training_algorithm        : Adam
verbosity                 : 20

