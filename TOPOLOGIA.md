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


O gerador é uma rede neural feedforward composta por várias camadas densas fully connected, camadas de Dropout, e camadas de ativação . O gerador recebe um vetor de ruído aleatório e um rótulo (classe) como entrada, gerando uma amostra como saída. Os rótulos condicionais são fornecidos ao modelo durante o treinamento. Ao treinar a CGAN, os dados reais são associados a seus respectivos rótulos, e os rótulos são passados para o gerador juntamente com os vetores de ruído aleatórios.
O gerador é projetado para receber tanto os vetores de ruído aleatórios (latent_input) quanto os rótulos condicionais (label_input) como entradas. Essa informação condicional é incorporada nas camadas do gerador e ajuda a influenciar a geração dos dados sintéticos.

Construção do Discriminador com Rótulos Condicionais: O discriminador é projetado para receber as amostras de dados reais ou sintéticas (neural_model_input) juntamente com os rótulos condicionais (label_input). Essa informação condicional é usada pelo discriminador para tentar distinguir entre dados reais e dados gerados pelo gerador, considerando também a classe associada a cada amostra.

Durante o treinamento da CGAN, a perda (loss) do discriminador e do gerador é calculada com base nos rótulos condicionais associados aos dados reais e gerados. A informação condicional ajuda a guiar o processo de treinamento, permitindo que o gerador aprenda a gerar amostras que se assemelham a cada classe específica e o discriminador aprenda a fazer distinções precisas entre as classes reais e falsas.

