


## Configuração do Experimento para camadas densas de 256. 

1. Comando utilizado para a execução do experimento:
   ```
     python main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam

   ```


 
### Configurações utilizadas:


  --------------------------------------------------------------

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
      

     

   
                      

    --------------------------------------------------------------



<div style="text-align: center;">
<table>
    <tbody>
        <tr>
           <td colspan="2" style="text-align: center;">
                <h2>Métricas de similaridade  </h2>
           </td>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/Comparison_Real_Synthetic_page_1.png" alt="" style="max-width:50%;"></td>
        </tr>
</div>




 [Log completo da campanha](https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/logging.log)
    

<table> 
    <tbody> 
        <tr>
            <td colspan="2" style="text-align: center;">
                <h2>Métricas de aplicabilidade RF </h2>
            </td>
        </tr>
        <tr>
            <td>
                <div style="position: absolute; top: 50px; left: 50px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/RandomForest_Synthetic_page_1.png"
                         alt="RandomForest Synthetic 256"
                         style="max-width: 160%;">
                </div>
            </td>
            <td>
                <div style="position: absolute; top: 50px; left: 200px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/RandomForest_Real_page_1.png"
                         alt="RandomForest Real 256"
                         style="max-width: 160%;">
                </div>
            </td>
        </tr>
    </tbody> 
</table>




   

  <table>
    <tbody> 
        <tr>
            <tr>
            <td colspan="2" style="text-align: center;">
                <h2>Matrizes de confusão RF </h2>
            </td>
        </tr>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/confusion_matrix/CM_Synthetic_RandomForest_k5_page_1.png" alt="" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/confusion_matrix/CM_Real_RandomForest_k5_page_1.png" alt="" style="max-width:160%;"></td>
        </tr>
    <tbody> 
        <tr>
            <th width="20%">Synthetic_RandomForest_k5_256</th>
            <th width="20%">Real_RandomForest_k5_256</th>
        </tr>
        <tr>



 <div style="text-align: center;">
<table>
    <tbody>
        <tr>
           <td colspan="2" style="text-align: center;">
                <h2> Curva de treinamento </h2>
           </td>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/training_curve/curve_training_error_k_5_page_1.png" alt="" style="max-width:50%;"></td>
        </tr>
</div>

