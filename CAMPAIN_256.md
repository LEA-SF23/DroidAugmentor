# Experimento 1: camadas densas de 256

Nas seções a seguir apresentamos:
- a execução (comando) e configuração do experimento;
- os gráficos das métricas de similaridade;
- os gráficos das métricas de aplicabilidade utilizando o KNN;
- as matrizes de confusão do Random Forest para as 5 dobras;
- o gráfico da curva de treinamento.
  


Conforme evidenciado, aprimorar a qualidade dos dados demanda um treinamento que empregue um maior número de camadas densas. Isso se reflete claramente nos gráficos que representam as métricas de similaridade, aplicabilidade e matrizes de confusão, os quais indicam um desempenho menos satisfatório do que os experimentos 2 e 3.
--------------------


Os logs completos e todos os gráficos gerados pela execução do experimento podem ser vistos nos dois links a seguir. Os logs completos incluem os gráficos dos 5 classificadores utilizados (Random Forest, Support Vector Machine, KNN, Decision Tree, AdaBoost) e todos os detalhes de saídas da execução em formato textual.

[Log completo da campanha](https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/logging.log)

[Relação completa de gráficos e dados do experimento](https://github.com/LEA-SF23/DroidAugmentor/tree/main/Campains_Results/256)

 
## Execução e Configuração 

1. Comando utilizado para a execução do experimento:
   ```
     python3 main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam

   ```


2. Configuração do experimento: 


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


## Métricas de Similaridade

As métricas de similaridade permitem verificar se os dados gerados são diferentes dos dados originais e, ao mesmo tempo, seguem o mesmo padrão estatístico.

![enter image description here](https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/Comparison_Real_Synthetic_page_1.png)


## Métricas de Aplicabilidade

As métricas de aplicabilidade são as convencionais, como Acurácia, Precisão, Recall e F1-Score. O objetivo é verificar se os classificadores são capazes de classificar os dados sintéticos de maneira similar aos dados reais. Em caso positivo, pode-se inferir que os dados sintéticos são realistas e adequados.



<table> 
    <tbody> 
            <td>
                <div style="position: absolute; top: 50px; left: 50px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/KNN_Synthetic_page_1.png"
                         alt="KNN_Synthetic 256"
                         style="max-width: 160%;">
                </div>
            </td>
            <td>
                <div style="position: absolute; top: 50px; left: 200px;">
                    <img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/KNN_Real_page_1.png"
                         alt="KNN_Real_256"
                         style="max-width: 160%;">
                </div>
            </td>
        </tr>
       <tbody> 
        <tr>
            <th width="20%">KNN_Synthetic 256</th>
            <th width="20%">KNN_Real 256</th>
        </tr>
    </tbody> 
</table>







   
## Matrizes de Confusão 

Ajuda a entender como o modelo está se saindo em relação a classificação das categorias de interesse, quanto mais parecidas com o real, maior é a qualidade dos dados.
  <table>
    <tbody> 
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/confusion_matrix/CM_Synthetic_KNN_k5_page_1.png" alt="" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/confusion_matrix/CM_Real_KNN_k5_page_1.png" alt="" style="max-width:160%;"></td>
        </tr>
    <tbody> 
        <tr>
            <th width="20%">Synthetic_KNN_k5_256</th>
            <th width="20%">Real_KNN_k5_256</th>
        </tr>
        <tr>
    </tbody> 
</table>



## Curva de Treinamento
Curva da cGAN do gerador e discriminador:A figura ilustra a dinâmica entre o gerador e o discriminador durante o processo de aprendizado. Ao longo das iterações, ocorre uma luta entre as duas redes. O gerador busca produzir amostras cada vez mais autênticas para enganar o discriminador, enquanto o discriminador se esforça para melhorar sua capacidade de distinguir entre real e falso. Essa competição leva a uma convergência, onde as amostras geradas se tornam quase indistinguíveis dos dados reais.
Alguns casos de não convergência em redes GAN podem ser eventualmente confirmados também através da observação do monitoramento das curvas de perda. As curvas de perda devem diminuir e estabilizar com o tempo. 

![enter image description here](https://github.com/LEA-SF23/DroidAugmentor/blob/main/Campains_Results/256/training_curve/curve_training_error_k_5_page_1.png)
