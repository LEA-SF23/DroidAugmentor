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

A CGAN (Conditional Generative Adversarial Network) é um modelo de rede neural feedforward composto por várias camadas densas fully connected, camadas de Dropout e camadas de ativação. Ele é projetado para gerar amostras condicionais a partir de um vetor de ruído aleatório e um rótulo (classe) como entrada.

Durante o treinamento, o gerador recebe tanto os vetores de ruído aleatórios (latent_input) quanto os rótulos condicionais (label_input). Essa informação condicional é incorporada nas camadas do gerador e influencia a geração dos dados sintéticos.

Por sua vez, o discriminador é projetado para receber amostras de dados reais ou sintéticas (neural_model_input) juntamente com os rótulos condicionais (label_input). Essa informação condicional é utilizada pelo discriminador para distinguir entre dados reais e dados gerados pelo gerador, levando em conta a classe associada a cada amostra.

No treinamento da CGAN, a perda (loss) do discriminador e do gerador é calculada com base nos rótulos condicionais associados aos dados reais e gerados. Essa informação condicional orienta o processo de treinamento, permitindo que o gerador aprenda a gerar amostras semelhantes a cada classe específica e que o discriminador aprenda a fazer distinções precisas entre as classes reais e falsas.
