# Experimento 1: camadas densas de 1024

Nas seções a seguir apresentamos:
- a execução (comando) e configuração do experimento;
- os gráficos das métricas de similaridade;
- os gráficos das métricas de aplicabilidade utilizando o Random Forest;
- as matrizes de confusão do Random Forest para as 5 dobras;
- o gráfico da curva de treinamento.

Os logs completos e todos os gráficos gerados pela execução do experimento podem ser vistos nos dois links a seguir. Os logs completos incluem os gráficos dos 5 classificadores utilizados (Random Forest, Support Vector Machine, KNN, Decision Tree, AdaBoost) e todos os detalhes de saídas da execução em formato textual.

[Log completo da campanha](https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/logging.log)

[Relação completa de gráficos e dados do experimento](https://github.com/LEA-SF23/DroidAugmentor/tree/main/Campains_Results/256)



## Configuração do Experimento para camadas densas de 1024. 

1. Comando utilizado para a execução do experimento:
   ```
    python main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_1024/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 1024 --dense_layer_sizes_d 1024 --number_epochs 1000 --training_algorithm Adam


   ```

<table> 
    <tbody> 
        <tr>
            <td colspan="2" style="text-align: center;">
                <h2>Métricas de aplicabilidade RF </h2>
                <p>Ao verificar se os classificadores são capazes de classificar os dados sintéticos de maneira similar aos dados reais, pode-se inferir que os dados sintéticos são realistas e adequados.</p>
            </td>
        </tr>
        <tr>
            <td>
                <div style="position: absolute; top: 50px; left: 50px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/1024/KNN_Synthetic_page_1.png"
                         alt="KNN_Synthetic 256"
                         style="max-width: 160%;">
                </div>
            </td>
            <td>
                <div style="position: absolute; top: 50px; left: 200px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/1024/KNN_Real_page_1.png"
                         alt="KNN_Real 256"
                         style="max-width: 160%;">
                </div>
            </td>
        </tr>
       <tbody> 
        <tr>
            <th width="20%">KNN_Synthetic 256</th>
            <th width="20%">KNN_Real  256</th>
        </tr>
    </tbody> 
</table>


       
  

 ### Configurações utilizadas:


  --------------------------------------------------------------

      activation_function       : LeakyReLU
      batch_size                : 32
      classifier                : ['RandomForest', 'SupportVectorMachine', 'KNN', 'DecisionTree', 'AdaBoost']
      data_type                 : float32
      dense_layer_sizes_d       : [[1024]]
      dense_layer_sizes_g       : [[1024]]
      dropout_decay_rate_d      : 0.4
      dropout_decay_rate_g      : 0.2
      initializer_deviation     : 0.02
      initializer_mean          : 0.0
      activation_function       : LeakyReLU
      batch_size                : 32
      classifier                : ['RandomForest', 'SupportVectorMachine', 'KNN', 'DecisionTree', 'AdaBoost']
      data_type                 : float32
      dense_layer_sizes_d       : [[1024]]
      dense_layer_sizes_g       : [[1024]]
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
      

--------------------------------------------------------------


 <table>
    <tbody> 
        <tr>
            <th width="20%">Synthetic_KNN_k5_1024</th>
            <th width="20%">Real_KNN_k5_1024</th>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/1024/confusion_matrix/CM_Synthetic_KNN_k5_page_1.png" alt="" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/1024/confusion_matrix/CM_Real_KNN_k5_page_1.png" alt="" style="max-width:160%;"></td>
        </tr>
    <tbody> 
        <tr>
            <th width="20%">Synthetic_KNN_k5_1024</th>
            <th width="20%">Real_KNN_k5_1024 </th>
        </tr>
        <tr>

