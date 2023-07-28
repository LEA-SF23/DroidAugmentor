# DroidAugmentor
Ferramenta de geração de dataset


Passos (versão simples):
1. git clone git@github.com:LEA-SF23/DroidAugmentor.git
2. cd DroidAugmentor
3. ./scripts/docker_build.sh
4. ./scripts/docker_run_shared_dir.sh . 
5. ./scripts/run_see_if_it_works.sh

## 

### Config with pipenv

```
pip install pipenv
```
```
pipenv install -r requirements2.txt
```
### run with pipenv
```
pipenv run python generation.py --input_dataset  "Datasets/drebin215_permissions_apiCalls_intents.csv" --data_type float32 --num_samples_class_malware 10000 --num_samples_class_benign 10000 --number_epochs 10000  --classifier knn --k_fold 5 --latent_dimension 128
```


### Running/ Executar

```
python generation.py --input_dataset  "Datasets/drebin215_permissions_apiCalls_intents.csv" --data_type float32 --num_samples_class_malware 10000 --num_samples_class_benign 10000 --number_epochs 10000  --classifier knn --k_fold 5 --latent_dimension 128
```

### Checklist de implementação

- [x] --input_dataset
- [x] --data_type
- [x] --output_dataset
- [x] --num_samples_class_malware
- [x] --num_samples_class_benign
- [x] --number_epochs
- [x] --classifier choices knn, perceptron, random_forest
- [x] --k_fold
- [x] --latent_dimension
- [x] --activation_function: string default: LeakyReLU Opções:ReLU, PreLU)
- [x] --training_algorithm : Define o algoritmo que será utilizado durante o treinamento das redes geradora e discriminadora. O algoritmo é responsável por atualizar os pesos das redes com base na função de perda durante o processo de aprendizado. default: Adam  Opções:('Adam', 'RMSprop', 'Adadelta')
- [x] --dropout_decay_rate_g: default=0.2, "Taxa de decaimento do dropout do gerador da cGAN"
- [x] --dropout_decay_rate_d  default=0.4, "Taxa de decaimento do dropout do discriminador da cGAN"  
- [x] --dense_layer_sizes_g default=[128, 256, 512]  Valor das camadas densas do gerador
- [x] --dense_layer_sizes_d default=[512, 256, 128]  Valor das camadas densas do discriminador
- [x] --use_gpu default=False  Opção para usar a GPU do TensorFlow
- [x] --batch_size: Tamanho do lote, default: 32. Parâmetro que determina o número de amostras de dados processadas em cada passo de treinamento da rede. É a quantidade de exemplos passados pela rede antes de realizar uma atualização dos pesos do modelo. Opções: (16, 32,64).
- [x] --output_format_plot, Default pdf. Seleção dos arquivos de saída do gráfico (PDF, PNG)
- [x] Verificar output_dataset
- [x] alterar help input e output arquivo do dataset de entrada, arquivo dataset de saida.
- [x] MSE
- [x] Cosine
- [x] Salvar gráfico, resultados e parâmetros de entrada em output_dataset

<<<<<<< O que falta


- [ ] Usar classificador para treinar com dados sintéticos e híbrido 3 resultados.
- [ ]  --output_format_plot remover.
- [ ] output_model
- [ ] Suprimir prints da cGAN no treinamento
- [ ] colocar no plot o nome do classificador.
- [ ] rodar perceptron.
- [ ] Medição do tempo em cada etapa do processo da ferramenta.
- [ ] Aumentar a variabilidade de datasets.
- [ ] Melhorar a rede com menos dados trabalho futuro.
- [ ] duas saidas, desemepenho do classificador. Matriz de confusão.
- [ ] colocar no github os parâmetros da ferramenta.

Parâmetros cGAN
latent_dim      128
epochs          10000
np.random seed  np.random seed
Loss            binary_crossentropy

Parâmetros Gerador

Dense                   128
Droupout                0,2
LeakyReLu               0,2           
BatchNormalization      0,8
Dense                   256
Droupout                0,2
LeakyReLu               0,2
BatchNormalization      0,8
Dense                   512
Droupout                0,2
LeakyReLu               0,2
BatchNormalization      0,8
Activation              Sigmoid


Parâmetros Discriminador
 
Dense                  512
LeakyReLu              0,2
Dense                  256
Droupout               0,4
Dense                  128
Activation            Sigmoid


      
      
      
      
      


      


