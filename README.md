# DroidAugmentor

EXPLICAR ALGO AQUI


## Topologia da rede neural

[Link]()


## Instalação e utilização 

1. Usar imagem disponivel no hub.docker.com
   ```
   
   ```
     
2. Construir uma imagem do docker. O dockerfile já está disponível no repositório e pode ser modificado com a necessidade do usuário.
   
    ```
   docker build -t IMAGE_NAME .
    ```
    
3. Instalar dependências e executar em um linux qualquer
    - Instalação dos [requirements](requirements.txt)
```
pip install pipenv
```
```
pipenv install -r requirements.txt
```
- Execução da ferramenta
```
pipenv run python main.py -i "dataset/seu.csv" -c knn -o --output_dir
```

## Automatizar os experimentos

A ferramenta conta com o run_campaign.py para automatizar a avaliação de algoritmos de aprendizado de máquina em diferentes configurações e conjuntos de dados. Ele executa várias campanhas de avaliação com diferentes parâmetros e registra os resultados em arquivos de log para análise posterior. O resultado final é uma análise comparativa das diferentes configurações de algoritmos em relação aos conjuntos de dados utilizados.

Com todas as dependências instaladas execute: 
```
python run_campaign.py
```


### Executar no Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

```
!pip install -r requirements.txt
```
```
input_file_path = "/content/seu.csv"
```

```
!python main.py -i "$input_file_path" -c knn -o --output_dir
```

Obs.: Lembre-se de ter Models, Tools e a main devidamente importada no seu drive.


 ## Parâmetros
    --------------------------------------------------------------
   
          (main.py):

           -i ,  --input_dataset        Caminho para o arquivo do dataset real de entrada         
           -c ,  --classifier           Classificador a ser utilizado     
           -o ,  --output_dir           Diretório para gravação dos arquivos de saída.
           --data_type                  Tipo de dado para representar as características das amostras.
           --num_samples_class_malware  Número de amostras da Classe 1 (maligno).
           --num_samples_class_benign   Número de amostras da Classe 0 (benigno).
           --number_epochs              Número de épocas (iterações de treinamento) da cGAN.
           --k_fold                     Número de subdivisões da validação cruzada 
           --initializer_mean           Valor central da distribuição gaussiana do inicializador.
           --initializer_deviation      Desvio padrão da distribuição gaussiana do inicializador.
           --latent_dimension           Dimensão do espaço latente para treinamento cGAN.
           --training_algorithm         Algoritmo de treinamento para cGAN. Opções: 'Adam', 'RMSprop', 'Adadelta'.
           --activation_function        Função de ativação da cGAN. Opções: 'LeakyReLU', 'ReLU', 'PReLU'.
           --dropout_decay_rate_g       Taxa de decaimento do dropout do gerador da cGAN.
           --dropout_decay_rate_d       Taxa de decaimento do dropout do discriminador da cGAN.
           --dense_layer_sizes_g        Valores das camadas densas do gerador.
           --dense_layer_sizes_d        Valores das camadas densas do discriminador.
           --use_gpu                    Opção para usar a GPU para treinamento.
           --batch_size                 Tamanho do lote da cGAN.
           --verbosity                  Nível de verbosidade.
           --save_models                Opção para salvar modelos treinados.
           --path_confusion_matrix      Diretório de saída das matrizes de confusão.
           --path_curve_loss            Diretório de saída dos gráficos de curva de treinamento.

        --------------------------------------------------------------
        

## Ambientes de teste







## Agradecimentos




This study was financed in part by the Coordenação
de Aperfeiçoamento de Pessoal de Nível Superior - Brasil
(CAPES) - Finance Code 001. 

