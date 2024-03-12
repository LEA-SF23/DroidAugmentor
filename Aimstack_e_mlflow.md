# AIMSTACK
 Aimstack é uma ferramenta open-source para o monitoramento de métricas e metadados
### Como instalar
``` pip3 install aim ```

``` pip install aim ```

Para instalar no pipenv 

``` pipenv install aim``` 

# MLFLOW
  Mflow é uma ferramenta que tem o proposito de auxiliar o processo auxiliar de aprendizado de maquina através do monitoramento
### Como instalar
``` pip3 install mlflow ```

``` pip install mlflow ```

Para instalar no pipenv 

``` pipenv install mlflow``` 

## Dependências

Python 3

Aim versão mais recentes em ambas as máquinas (Versão mínima viável para uso do servidor 3.4.0)

Mflow mesma versão em ambas as máquinas

Mesma versão do Openssl entre as máquinas

Firewall precisa permitir transferência na porta ex: ``` sudo ufw allow 53800```


# Como instrumentalizar
## Aimstack
 Importar a biblioteca
```bash
$ import aim
$ from aim import Run

```
 
 Inicializar IP e porta utilizados para exemplificar
```bash
$ aim_run=Run(repo='aim://127.1.1.1:53800',experiment="test-keys")
```
Monitorar uma métrica
```bash
$  aim_run.track(métrica_a_ser_monitorada, name='nome_para_salvar_a_métrica')
```
## Mlflow
 Importar a biblioteca
```bash
$  import mlflow

```
No lado do cliente é necessario apontar o IP  e a porta do servidor
```bash
$ mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
```
Nome do experimento
```bash
$ mlflow.set_experiment("droid_augmentor")
```
```bash
$ mlflow.start_run()
```
Como monitorar uma métrica
```bash
$ mlflow.log_metric('nome',métrica)
```
# Gerar certificados

Primeiro é necessaria a geração de certificados e chaves para serem utilizados nas execuções
-opcional criar um arquivo de configuração


1.Gerar a chave
```bash
$  openssl genrsa -out nome_da_chave.key 2048
```
2.Gerar arquivo .pem
```bash
$ openssl req -config test.conf -new -out nome_pem.csr.pem
```
3.Gerar o certificado 
```bash
$ openssl x509 -req -days 365 -extfile dev.config -extensions v3_req -in nome_pem.csr.pem -signkey nome_da_chavae.key -out nome_do_certficado.crt
```
3.4 Opcional gerar um .pfx
```bash
$ openssl pkcs12 -export -out arquivo.pfx -inkey nome_da_chave.key -in nome_cert.crt -password pass:$'senha'
```

3.5 Alternativamente quando não utilizado arquivo de configuração
```bash
$ openssl req -x509 -nodes -days 365 -out nome_pen.csr.pem -subj "/CN=example.com" -addext "subjectAltName = IP:número de ip"
```
4.Exportar o crt para o cliente e exportar a variável de ambiente no cliente


``` export __AIM_CLIENT_SSL_CERTIFICATES_FILE__= nome_do_certificado.crt  ```


5.Inicializar o servidor
```bash
$ aim server --repo diretorio para ser salvo  -h ip --ssl-keyfile nome_da_chave.key --ssl-certfile nome_do_certificado.crt 
```
6.Executar o código no cliente 

7.Visualização no servidor
```bash
$ aim up
```

###  Parâmetros disponíveis:
Caminho para diretorio pai do repo .aim. Por padrão é utilizado o diretorio atual
```
--repo <repo_path>
```

Ip a ser utilizado para estabelecer o servidor
```
-h &#124; --host <host>
```

Porta a ser utilizada para estabelecer o servidor, por padrão utilzada 53800
```
-p &#124; --port <port>
```	

Especifica o número de gPRC workers. Por padrão é  1 worker.
```	
-w &#124; --workers <N>
```	

Especifica o caminho para o arquivo chave 
```	
--ssl-keyfile
```		

Especifica o caminho para o certificado
```	
--ssl-certfile
```		

Especifica o nivel log para o pacote python logging package. Por padrão ``WARNING``
```		
--log-level
```		
