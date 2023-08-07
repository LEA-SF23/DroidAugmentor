# DroidAugmentor

Ferramenta de treinamento e avaliação de cGANs para geração de dados sintéticos


## Topologia da rede neural

[Link]()


## Instalação e utilização 

1. Usar imagem disponivel no hub.docker.com
   ```
   docker_run.sh
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

A ferramenta conta com o run_campaign.py para automatizar a avaliação, executando várias campanhas de avaliação com diferentes parâmetros e registra os resultados em arquivos de log para análise posterior. O resultado final é uma análise comparativa das diferentes configurações em relação aos conjuntos de dados utilizados.

Com todas as dependências instaladas execute: 
```
python run_campaign.py
```
Exemplo de execução de campanha:
```
run_campaign.py -c sf23_1l_256

```
Mesma campanha sendo executada por main.py:
```
pipenv run python main.py --verbosity 20 --output_dir outputs/out_2023-08-05_12-04-18/sf23_1l_256/combination_2 --input_dataset datasets/drebin215_original_5560Malwares_6566Benign.csv --dense_layer_sizes_g 256 --dense_layer_sizes_d 256 --number_epochs 1000 --training_algorithm Adam
```
###  Parâmetros dos testes automatizados:

      --------------------------------------------------------------

    --campaign ou -c:    Especifica a campanha de avaliação que você deseja executar. 
                         Você pode fornecer o nome de uma campanha específica ou uma  
                         lista de campanhas separadas por vírgula. 
                         Por exemplo: --campaign sf23_1l_64 ou --campaign 
                         sf23_1l_64,sf23_1l_128.

    --demo ou -d:
                         Ativa o modo demo. Quando presente, o script será executado 
                         no modo demo, o que pode ter comportamento reduzido 
                         ou exibir informações de teste.
                         --verbosity ou -v: Especifica o nível de verbosidade do log.
                         Pode ser INFO (1) ou DEBUG (2). 
                         Por padrão, o nível de verbosidade é definido como INFO.


     Outros parâmetros de entrada são definidos dentro das campanhas de avaliação em 
     campaigns_available. Cada campanha tem suas próprias configurações específicas, 
     como input_dataset, number_epochs, training_algorithm, dense_layer_sizes_g, 
     dense_layer_sizes_d, classifier, activation_function, dropout_decay_rate_g, 
     dropout_decay_rate_d, e data_type. As configurações podem variar dependendo do 
     objetivo e das configurações específicas de cada campanha.  


     Em campaigns_available o script irá iterar sobre as combinações de configurações 
     especificadas e executar os experimentos correspondentes.

    --------------------------------------------------------------


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


 ## Parâmetros da ferramenta:
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

A ferramenta foi testada e utilizada na prática nos seguintes ambientes:

1. Windows 10.
   Kernel Version = 10.0.19043
   Versão Python e bibliotecas conforme [requirements](requirements.txt).
   
2. Linux Ubuntu 22.04.2 LTS
   Kernel Version = 5.15.109+
   Versão Python e bibliotecas conforme [requirements](requirements.txt).

   
   
  





## Agradecimentos




Este estudo foi parcialmente financiado pela Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Código Financeiro 001.

