# Análise das características dos dados.


[Imagens completas](https://github.com/LEA-SF23/DroidAugmentor/tree/main/Features)
<div style="text-align: center;">
<table>
    <tbody>
        <tr>
           <td colspan="2" style="text-align: center;">
                <h2> Mapa de calor</h2>
              <p> Os gráficos demonstram que os dados sintéticos incorporaram 
padrões similares aos dados reais, tanto para amostras classificadas como malignas 
(M.) quanto para aquelas classificadas como benignas (B.). </p>
           </td>
        </tr>
        <tr>
            <td><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Features/heatmap.PNG" alt="" style="max-width:50%;"></td>
        </tr>
</div>

<table>
    <tbody>
        <tr>
            <td colspan="3" style="text-align: center;">
                <h2>Importância das Features</h2>
                <p> O gráfico que mostra as importâncias das diferentes características (features) do RandomForest treinado nos dados sintéticos e reais. Esse gráfico pode ajudar a identificar quais características têm mais impacto nas decisões do classificador. Como pode se observar, as features nos dados sintéticos mantiveram um padrão considerável a dos dados originais. </p>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;"><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Features/Sintetico_5560Malwares_6566Benign.png" alt="" style="max-width: 100%; height: auto;"></td>
            <td style="text-align: center;"><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Features/Drebin215_original_5560Malwares_6566Benign.png" alt="" style="max-width: 100%; height: auto;"></td>
            <td style="text-align: center;"><img src="https://github.com/LEA-SF23/DroidAugmentor/blob/main/Features/Sintetico_6063Malwares_6063Benign.png" alt="" style="max-width: 100%; height: auto;"></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th width="33.33%">Sintético 12.126 amostras desbalanceado</th>
            <th width="33.33%"> Debrin-215 dados reais  </th>
            <th width="33.33%">intético 12.126 amostras balanceado</th>
        </tr>
    </tbody>
</table>
