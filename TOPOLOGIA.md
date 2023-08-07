## Topologia da rede neural

A topologia da rede neural da cGAN pode ser observada nas figuras abaixo.


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


O gerador é uma rede neural feedforward composta por várias camadas densas fully connected, camadas de Dropout, camadas de ativação e camadas de normalização batch. O gerador recebe um vetor de ruído aleatório e um rótulo (classe) como entrada, gerando uma amostra como saída. O rótulo é incorporado no gerador por meio de uma camada, que mapeia o rótulo para um vetor de tamanho \textit{latent dim}.
