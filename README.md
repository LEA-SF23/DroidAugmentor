# DroidAugmentor: Ferramenta de treinamento e avaliação de cGANs para geração de dados sintéticos

EXPLICAR ALGO AQUI


## Topologia da rede neural

EXPLICAR ALGO AQUI

<table>
    <tbody> 
        <tr>
            <th width="7%">Gerador</th>
            <th width="10%">Discriminador</th>
        </tr>
        <tr>
             <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/model_generator.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/model_discriminator.png" alt="2023-08-07 4 40 06" style="max-width:100%;"></td>
        </tr>

</table>
## Experimental Evaluation

### Análise de treinamento

EXPLICAR ALGO AQUI

<table>
    <tbody> 
        <tr>
            <th width="20%">Fold 1</th>
            <th width="20%">Fold 2</th>
            <th width="20%">Fold 3</th>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_1.png" alt="2023-08-07 4 33 16" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_2.png" alt="2023-08-07 4 40 06" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_3.png" alt="2023-08-07 4 43 02" style="max-width:160%;"></td>
        </tr>
    <tbody> 
        <tr>
            <th width="20%">Fold 1</th>
            <th width="20%">Fold 2</th>
            <th width="20%">Fold 3</th>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_1.png" alt="2023-08-07 4 33 16" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_2.png" alt="2023-08-07 4 40 06" style="max-width:160%;"></td>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/layout/curve_training_error_k_3.png" alt="2023-08-07 4 43 02" style="max-width:160%;"></td>
        </tr>

</table>

###  Parameter Sensitivity Analysis

Parameter sensitivity of Conv. topology withuniform probabilistic injected failure Fprob =10%
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/sens_conv.png" alt="2018-06-04 4 33 16" style="max-width:50%;"></td>
        </tr>


</table>


### Comparing our Neural Networks
Comparison of topologies Dense (DE), LSTM (LS), and Convolutional (CO) for probabilistic injected failure and monitoring injected failure.
<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Injected Failure</th>
            <th width="10%">Monitoring Injected Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_pif.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_mif.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
        </tr>
        
</table>

### Comparison with the State-of-the-Art (Convolutional vs Probabilistic)

Comparison between the best neural network model and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional vs Probabilistic</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/results.png" alt="2018-06-04 4 33 16" style="max-width:120%;"></td>
        </tr>


</table>

### Qualitative Analysis

Impact, in terms of number (left) and duration (right) of a trace (S1) failed (Fmon = 20) and regenerated using the proposed BB-based (topology=Conv., threshold α =0.50, arrangements A =8, squared window width W = H =256) and prior probabilistic-based (threshold α =0.75).

<table>
    <tbody> 
        <tr>
            <th width="10%">Sessions Duration</th>
            <th width="10%">Number Sessions</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_duration.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_number_sessions.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
        </tr>
        
</table>

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


### Input parameters:

    Arguments(run_TNSM.py):
        
       -h, --help            Show this help message and exit
       --append, -a          Append output logging file with analysis results
       --demo, -d            Demo mode (default=False)
       --trials, -r          Mumber of trials (default=1)
       --start_trials,-s     Start trials (default=0)
       --skip_train, -t      Skip training of the machine learning model training?
       --campaign -c         Campaign [demo, mif, pif] (default=demo)
       --verbosity, -v       Verbosity logging level (INFO=20 DEBUG=10)


    --------------------------------------------------------------
   
    Arguments(main.py):

           -i ,  --input_dataset         
           -c ,  --classifier      
           -o ,  --output_dir        
           --data_type
           --num_samples_class_malware
           --num_samples_class_benign
           --number_epochs
           --k_fold 
           --initializer_mean 
           --initializer_deviation
           --latent_dimension 
           --training_algorithm
           --activation_function 
           --dropout_decay_rate_g
           --dropout_decay_rate_d 
           --dense_layer_sizes_g
           --dense_layer_sizes_d
           --use_gpu
           --batch_size
           --output_format_plot
           --verbosity
           --save_models
           --path_confusion_matrix 
           --path_curve_loss 
           
           
           
           
           
           
           
           

            
         

        --------------------------------------------------------------
        



## Requirements:

`matplotlib 3.4.1`
`tensorflow 2.4.1`
`tqdm 4.60.0`
`numpy 1.18.5`

`keras 2.4.3`
`setuptools 45.2.0`
`h5py 2.10.0`





## Complementary Results

### Comparison with the State-of-the-Art (Dense vs Probabilistic)
Comparison between the neural network Dense and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.

<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_pif_dense_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>

<table>
    <tbody> 
        <tr>
            <th width="10%">Monitoring Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_mif_dense_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>


### Comparison with the State-of-the-Art (LSTM vs Probabilistic) 

Comparison between the neural network LSTM and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.

<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_pif_lstm_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>

<table>
    <tbody> 
        <tr>
            <th width="10%">Monitoring Inject Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_mif_lstm_prob.png" alt="2023-03-16 4 33 16" style="max-width:100%;"></td>
        </tr>
</table>


## ACKNOWLEDGMENTS


This study was financed in part by the Coordenação
de Aperfeiçoamento de Pessoal de Nível Superior - Brasil
(CAPES) - Finance Code 001. We also received funding from
Rio Grande do Sul Research Foundation (FAPERGS) - Grant
ARD 10/2020 and Nvidia – Academic Hardware Grant

