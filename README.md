# DroidAugmentor
Ferramenta de geração de dataset

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

<<<<<<< O que falta
- [ ] batch_size: Tamanho do lote, default: 32. Parâmetro que determina o número de amostras de dados processadas em cada passo de treinamento da rede. É a quantidade de exemplos passados pela rede antes de realizar uma atualização dos pesos do modelo. Opções: (16, 32,64).
- [ ] output_model


