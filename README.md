# DroidAugmentor: Ferramenta de treinamento e avaliação de cGANs para geração de dados sintéticos

EXPLICAR ALGO AQUI


## Topologia da rede neural

EXPLICAR ALGO AQUI


## Steps to Install:

1. Upgrade and update
    - sudo apt-get update
    - sudo apt-get upgrade 
    
2. Installation of application and internal dependencies
    - git clone [https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network]
    - pip install -r requirements.txt
    
3. Test installation:
    - python3 main.py -h


## Run experiments:

###  Run (all F_prob experiments)
`python3 run_jnsm_mif.py -c lstm`

### Run (only one F_prob scenario)
`python3 main.py`

###  Run (all F_mon experiments)
`python3 run_mif.py -c lstm`

### Run (only one F_mon scenario)
`python3 main_mif.py`

### Config with pipenv

```
pip install pipenv
```
```
pipenv install -r requirements.txt
```

### Run with pipenv
```
pipenv run python main.py -i "dataset/seu.csv" -c knn -o --output_dir
```


### Running/ Executar no Google Colab

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



    --------------------------------------------------------------
   
    Arguments(main.py):

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
        



## Requirements:

`-i https://pypi.org/simple`
`absl-py 1.4.0; python_version >= '3.6'`
`astunparse 1.6.3`
`cachetools 5.3.1; python_version >= '3.7'`
`certifi 2023.7.22; python_version >= '3.6'`
`charset-normalizer 3.2.0; python_full_version >= '3.7.0'`
`contourpy 1.1.0; python_version >= '3.8'`
`cycler 0.11.0; python_version >= '3.6'`
`flatbuffers 1.12`
`fonttools 4.41.1; python_version >= '3.8'`
`gast 0.4.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'`
`google-auth 2.22.0; python_version >= '3.6'`
`google-auth-oauthlib 0.4.6; python_version >= '3.6'`
`google-pasta 0.2.0`
`grpcio 1.56.2; python_version >= '3.7'`
`h5py 3.9.0; python_version >= '3.8'`
`idna 3.4; python_version >= '3.5'`
`importlib-metadata 6.8.0; python_version < '3.10'`
`importlib-resources 6.0.0; python_version < '3.10'`
`joblib 1.3.1; python_version >= '3.7'`
`kaleido 0.2.1`
`keras 2.9.0`
`keras-preprocessing 1.1.2`
`kiwisolver 1.4.4; python_version >= '3.7'`
`libclang 16.0.6`
`markdown 3.4.4; python_version >= '3.7'`
`markupsafe 2.1.3; python_version >= '3.7'`
`matplotlib 3.7.2`
`numpy 1.21.5`
`oauthlib 3.2.2; python_version >= '3.6'`
`opt-einsum 3.3.0; python_version >= '3.5'`
`packaging 23.1; python_version >= '3.7'`
`pandas 1.4.4`
`pillow 10.0.0; python_version >= '3.8'`
`plotly 5.6.0`
`protobuf 3.19.6; python_version >= '3.5'`
`pyasn1 0.5.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'`
`pyasn1-modules 0.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'`
`pyparsing 3.0.9; python_full_version >= '3.6.8'`
`python-dateutil 2.8.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'`
`pytz 2023.3`
`requests 2.31.0; python_version >= '3.7'`
`requests-oauthlib 1.3.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'`
`rsa 4.9; python_version >= '3.6' and python_version < '4'`
`scikit-learn 1.1.1`
`scipy 1.10.1; python_version < '3.12' and python_version >= '3.8'`
`setuptools 68.0.0; python_version >= '3.7'`
`six 1.16.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'`
`tenacity 8.2.2; python_version >= '3.6'`
`tensorboard 2.9.1; python_version >= '3.6'`
`tensorboard-data-server 0.6.1; python_version >= '3.6'`
`tensorboard-plugin-wit 1.8.1`
`tensorflow 2.9.1`
`tensorflow-estimator 2.9.0; python_version >= '3.7'`
`tensorflow-io-gcs-filesystem 0.32.0; python_version < '3.12' and python_version >= '3.7'`
`termcolor 2.3.0; python_version >= '3.7'`
`threadpoolctl 3.2.0; python_version >= '3.8'`
`typing-extensions 4.7.1; python_version >= '3.7'`
`urllib3 1.26.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4, 3.5'`
`werkzeug 2.3.6; python_version >= '3.8'`
`wheel 0.41.0; python_version >= '3.7'`
`wrapt 1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'`
`zipp 3.16.2; python_version < '3.10'`




## ACKNOWLEDGMENTS


This study was financed in part by the Coordenação
de Aperfeiçoamento de Pessoal de Nível Superior - Brasil
(CAPES) - Finance Code 001. We also received funding from
Rio Grande do Sul Research Foundation (FAPERGS) - Grant
ARD 10/2020 and Nvidia – Academic Hardware Grant

