#!/usr/bin/python3
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import sys

from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']
# rcParams['font.family'] = 'serif'
# rcParams['font.sans-serif'] = ['Times']

FONT_NORMAL = 14
FONT_big = 14
FONT_BIG = 15
rcParams['font.size'] = FONT_NORMAL
SHOW = True

REAL ='Real'
SYNT ='Sintético'

RF = 'Random Forest'
SVM = 'SVM'
KNN = 'KNN'
DT = 'Decision Tree'
AB = 'AdaBoost'

AC = 'Acurácia'
PR = 'Precisão'
RE = 'Recall'
F1 = 'F1'

# L1N256 = "2 camadas,\n256 neurônios"
# L1N1024 = "2 camadas,\n1024 neurônios"
# L1N4096 = "2 camadas,\n4096 neurônios"
L1N256 = "256 neurônios"
L1N1024 = "1024 neurônios"
L1N4096 = "4096 neurônios"

L2 = "3 Camadas"


MSE = 'Erro Médio Quadrático'
CS = 'Similaridade de Cossenos'
KL = 'Divergência KL'
MM = 'Máx. Discrepância Média'


 
def auto_label(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > .86:
            #print("height:", height)
            ax.annotate("."+'{:.2f}'.format(height).split('.')[1],
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(-2, +1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONT_big)
        else:
            # print("height:", height)
            ax.annotate("."+'{:.2f}'.format(height).split('.')[1],
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(-2, +3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=FONT_big)


def plot_2bars(arquivo, titulo, labels, xlabel, sufixes, plota_linha, figheight, figwidth,
          accuracy_means, precision_means, recall_means, f1_means,
          accuracy_error, precision_error, recall_error, f1_error,
            accuracy_means_det=None, precision_means_det=None, recall_means_det=None, f1_means_det=None,
            accuracy_error_det=None, precision_error_det=None, recall_error_det=None, f1_error_det=None):

    mapa_cor = plt.get_cmap('tab10')  # carrega tabela de cores conforme dicionário
    mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
    mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
    cores = [mapa_escalar.to_rgba(i*2) for i in range(10)]
    width = 0.20  # the width of the bars


    plt.rc('font', size=FONT_BIG)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_BIG)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_BIG)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_BIG)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_BIG)  # fontsize of the tick labels
    plt.rc('figure', titlesize=FONT_BIG)  # fontsize of the figure title

    if len(accuracy_means)<= 3:
        plt.rc('legend', fontsize=FONT_NORMAL)  # legend fontsize
    else:
        plt.rc('legend', fontsize=FONT_BIG)  # legend fontsize

    if accuracy_means_det is not None:
        mapa_cor = plt.get_cmap('tab20c')  # carrega tabela de cores conforme dicionário
        mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
        mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
        cores = [mapa_escalar.to_rgba(i) for i in range(20)]
        #width = 0.10  # the width of the bars

    x = np.arange(len(labels))  # the label locations
    if accuracy_means_det is not None:
        x = np.arange(len(labels))*2


    fig, ax = plt.subplots()

    if figheight is not None:
        figheight *= 0.393701  # cm to inches
        fig.set_figheight(figheight)


    if figwidth is not None:
        figwidth *= 0.393701  # cm to inches
        fig.set_figwidth(figwidth)

    if plota_linha is not None:
        print("plota_linha: {}".format(plota_linha))
        acuracia_det = plota_linha['acuracia']  # 0.8611
        plt.axline((0, acuracia_det), (2.25, acuracia_det), color=cores[0])  # acurácia de referência
        precisao_det = plota_linha['precisao']  # 0.9445
        plt.axline((0, precisao_det), (2.25, precisao_det), color=cores[1])  # precisão de referência
        recall_det = plota_linha['recall']  # 0.4899
        plt.axline((0, recall_det), (2.25, recall_det), color=cores[2])  # recall de referência
        f1_det = plota_linha['f1']  # 0.6451
        plt.axline((0, f1_det), (2.25, f1_det), color=cores[3])  # f1 de referência

    if accuracy_means_det is None:
        rects1 = ax.bar(x - width*2+width/2, accuracy_means, yerr=accuracy_error, width=width, color=cores[0], label=AC)
        rects2 = ax.bar(x - width*1+width/2, precision_means, yerr=precision_error, width=width, color=cores[1], label=PR)
        rects3 = ax.bar(x + width*0+width/2, recall_means, yerr=recall_error, width=width, color=cores[2], label=RE)
        rects4 = ax.bar(x + width*1+width/2, f1_means, yerr=f1_error, width=width, color=cores[3], label=F1)

    else:
        index_2nd_color = 0  #DENSE
        if sufixes[1] == "PR": #Probabilistc algorithm
            index_2nd_color = 1

        rects1 = ax.bar(x - width * 4 + width / 2, accuracy_means, yerr=accuracy_error, width=width,
                        color=cores[0], label='{} {}'.format(AC, sufixes[0]))
        rects1d = ax.bar(x - width * 3 + width / 2, accuracy_means_det, yerr=accuracy_error_det, width=width,
                         color=cores[2+index_2nd_color], label='{} {}'.format(AC, sufixes[1]))

        rects2 = ax.bar(x - width * 2 + width / 2, precision_means, yerr=precision_error, width=width,
                        color=cores[4], label='{} {}'.format(PR, sufixes[0]))
        rects2d = ax.bar(x - width * 1 + width / 2, precision_means_det, yerr=precision_error_det, width=width,
                         color=cores[6+index_2nd_color], label='{} {}'.format(PR, sufixes[1]))

        rects3 = ax.bar(x + width * 0 + width / 2, recall_means, yerr=recall_error, width=width,
                        color=cores[8], label='{} {}'.format(RE, sufixes[0]))
        rects3d = ax.bar(x + width * 1 + width / 2, recall_means_det, yerr=recall_error_det, width=width,
                         color=cores[10+index_2nd_color], label='{} {}'.format(RE, sufixes[1]))

        rects4 = ax.bar(x + width * 2 + width / 2, f1_means, yerr=f1_error, width=width,
                        color=cores[12], label='{} {}'.format(F1, sufixes[0]))
        rects4d = ax.bar(x + width * 3 + width / 2, f1_means_det, yerr=f1_error_det, width=width,
                         color=cores[14+index_2nd_color], label='{} {}'.format(F1, sufixes[1]))


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)#, fontsize=FONT_BIG)
    ax.set_title(titulo)#, fontsize=FONT_BIG)
    for i in range(len(x)):
        x[i] = x[i] + 0.0

    ax.set_xticks(x)
    print("x      {}".format(x))
    ax.set_xticklabels(labels)
    print("labels {}".format(labels))
    ax.set_yticks([0.0, 0.5, 1.0])
    #ax.set_ylabel("Média e desv. padrão p/ 5 dobras")
    ax.set_ylim(0.0, 1.2)
    ax.legend(loc='lower center', ncol=4, framealpha=1.0)
    #ax.set_xlim(0.5, 9.5)



    auto_label(ax, rects1)
    auto_label(ax, rects2)
    auto_label(ax, rects3)
    auto_label(ax, rects4)

    if accuracy_means_det is not None:
        auto_label(ax, rects1d)
        auto_label(ax, rects2d)
        auto_label(ax, rects3d)
        auto_label(ax, rects4d)
        #plt.axes([0.0, 15.0, 0.0, 1.0])
        pass


    fig.tight_layout()

    #plt.show()

    plt.savefig(arquivo, dpi=300)

 

def plot_1bar(arquivo, titulo, labels, xlabel, sufixes, plota_linha, figheight, figwidth,
              CS_means, MSE_means, KL_means, MM_means,
              CS_error, MSE_error, KL_error, MM_error):

    mapa_cor = plt.get_cmap('tab10')  # carrega tabela de cores conforme dicionário
    mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
    mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final
    cores = [mapa_escalar.to_rgba(i*2) for i in range(10)]
    width = 0.30  # the width of the bars


    plt.rc('font', size=FONT_NORMAL)  # controls default text sizes
    plt.rc('axes', titlesize=FONT_NORMAL)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_NORMAL)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_NORMAL)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_NORMAL)  # fontsize of the tick labels
    plt.rc('figure', titlesize=FONT_NORMAL)  # fontsize of the figure title

    # if len(accuracy_means)<= 3:
    #     plt.rc('legend', fontsize=FONT_NORMAL)  # legend fontsize
    # else:
    #     plt.rc('legend', fontsize=FONT_BIG)  # legend fontsize


    x = np.arange(len(labels))  # the label locations



    fig, ax = plt.subplots()

    if figheight is not None:
        figheight *= 0.393701  # cm to inches
        fig.set_figheight(figheight)


    if figwidth is not None:
        figwidth *= 0.393701  # cm to inches
        fig.set_figwidth(figwidth)

    #rects1 = ax.bar(x - width*2+width/2, CS_means, yerr=CS_error, width=width, color=cores[3], label=CS)
    rects2 = ax.bar(x - width*1+width/2, MSE_means, yerr=MSE_error, width=width, color=cores[0], label=MSE)
    rects3 = ax.bar(x + width*0+width/2, KL_means, yerr=KL_error, width=width, color=cores[1], label=KL)
    rects4 = ax.bar(x + width*1+width/2, MM_means, yerr=MM_error, width=width, color=cores[2], label=MM)



    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(xlabel)#, fontsize=FONT_BIG)
    ax.set_title(titulo)#, fontsize=FONT_BIG)
    for i in range(len(x)):
        x[i] = x[i] + 0.0

    ax.set_xticks(x)
    print("x      {}".format(x))
    ax.set_xticklabels(labels)
    print("labels {}".format(labels))
    ax.set_yticks([0.0, 0.5, 1.0])
    #ax.set_ylabel("Média e desv. padrão p/ 5 dobras")
    #ax.set_ylim(0.5,1)
    ax.legend(loc='upper center', ncol=4, framealpha=1.0)
    #ax.set_xlim(0.5, 9.5)

    #auto_label(ax, rects1)
    auto_label(ax, rects2)
    auto_label(ax, rects3)
    auto_label(ax, rects4)



    fig.tight_layout()

    #plt.show()

    plt.savefig(arquivo, dpi=300)


def main():
    global FONT_NORMAL, FONT_BIG
    #errors = load_mif_errors()


    plota_linha = None

    title = None
    plots = []





    #L3 = "3 layers" #deu problema
    data = {}
    data[REAL] = {}
    data[SYNT] = {}
    labels2 = [L1N256, L1N1024, L1N4096]
    # DREBIN TOPOLOGIAS KNN
    data[REAL][L1N256] = {}
    data[REAL][L1N256][AC] =  [0.9820419022281344, 0.9750581975390755, 0.9750581975390755, 0.9723977386099102, 0.9607582307948122]
    data[REAL][L1N256][PR] =  [0.988909426987061, 0.9823255813953489, 0.9770009199632015, 0.9606087735004476, 0.9618959107806692]
    data[REAL][L1N256][RE] =  [0.9622302158273381, 0.9496402877697842, 0.9550359712230215, 0.9649280575539568, 0.9307553956834532]
    data[REAL][L1N256][F1] =  [0.9753874202370101, 0.9657064471879286, 0.965893587994543, 0.9627635711081202, 0.9460694698354662]

    data[SYNT][L1N256] = {}
    data[SYNT][L1N256][AC] =  [0.9930162953109412, 0.43997339541070835, 0.8041237113402062, 0.0029930162953109413, 0.48054539408047886]
    data[SYNT][L1N256][PR] =  [0.9814651368049426, 0.3977110157367668, 0.6537330981775427, 0.0, 0.4158563949139865]
    data[SYNT][L1N256][RE] =  [1.0, 1.0, 1.0, 0.0, 1.0]
    data[SYNT][L1N256][F1] =  [0.9906458797327394, 0.5690890481064483, 0.7906150017774618, 0.0, 0.5874273639725303]

    #1024
    data[REAL][L1N1024] = {}
    data[REAL][L1N1024][AC] =   [0.9916860658463585, 0.9830395743265713, 0.9803791153974061, 0.9767209843698038, 0.9670768207515796]
    data[REAL][L1N1024][PR] =  [0.9954421148587056, 0.9871441689623508, 0.9782016348773842, 0.9618794326241135, 0.9651056014692379]
    data[REAL][L1N1024][RE] = [0.9820143884892086, 0.966726618705036, 0.9685251798561151, 0.9757194244604317, 0.9451438848920863]
    data[REAL][L1N1024][F1] =  [0.9886826618379357, 0.9768287142208089, 0.973339358337099, 0.9687500000000001, 0.9550204452521581]


    data[SYNT][L1N1024] = {}
    data[SYNT][L1N1024][AC] = [0.9228466910542068, 0.9421350182906552, 0.9038909211839042, 0.8992351180578649, 0.9348187562354506]
    data[SYNT][L1N1024][PR] = [0.9751619870410367, 0.9746963562753036, 0.9194699286442406, 0.9089989888776542, 0.9561752988047809]
    data[SYNT][L1N1024][RE] = [0.8120503597122302, 0.8660071942446043, 0.8111510791366906, 0.8084532374100719, 0.8633093525179856]
    data[SYNT][L1N1024][F1] = [0.886162904808636, 0.9171428571428571, 0.8619206880076445, 0.8557829604950022, 0.9073724007561437]


    #L1N4096
    data[SYNT][L1N4096] = {}
    data[SYNT][L1N4096][AC] =  [0.9181908879281676, 0.928500166278683, 0.9135350848021284, 0.9075490522115065, 0.9231792484203525]
    data[SYNT][L1N4096][PR] = [0.9811111111111112, 0.9901639344262295, 0.9650655021834061, 0.9417372881355932, 0.9651531151003168]
    data[SYNT][L1N4096][RE] = [0.7940647482014388, 0.814748201438849, 0.7949640287769785, 0.7994604316546763, 0.8219424460431655]
    data[SYNT][L1N4096][F1] = [0.8777335984095428, 0.8939319190922546, 0.8717948717948718, 0.8647859922178988, 0.8878096163186012]

    data[REAL][L1N4096] = {}
    data[REAL][L1N4096][AC] = [0.9890256069171932, 0.9820419022281344, 0.9780512138343864, 0.9780512138343864, 0.962753574991686]
    data[REAL][L1N4096][PR] = [0.9882352941176471, 0.9826642335766423, 0.9694793536804309, 0.9611992945326279, 0.9545454545454546]
    data[REAL][L1N4096][RE] = [0.9820143884892086, 0.9685251798561151, 0.9712230215827338, 0.9802158273381295, 0.9442446043165468]
    data[REAL][L1N4096][F1] = [0.9851150202976996, 0.9755434782608696, 0.9703504043126684, 0.9706144256455922, 0.9493670886075949]
    
    # #
    # data[SYNT][L2] = {}
    # data[SYNT][L2][AC] =[0.6301962088460259, 0.6301962088460259, 0.6301962088460259, 0.6301962088460259, 0.6301962088460259]
    #
    # data[SYNT][L2][PR] =[0.729903536977492, 0.7343880099916736, 0.7757575757575758, 0.5, 0.9463087248322147]
    # data[SYNT][L2][RE] =[0.8165467625899281, 0.7931654676258992, 0.5755395683453237, 0.006294964028776978, 0.12679856115107913]
    # data[SYNT][L2][F1] =[0.7707979626485569, 0.7626459143968872, 0.6608156943727412, 0.012433392539964477, 0.22363203806502777]
    #
    # data[REAL][L2] = {}
    # data[REAL][L2][AC] =[0.983372131692717, 0.9807116727635518, 0.9810442301296973, 0.9707349517791819, 0.9610907881609577]
    # data[REAL][L2][PR] =[0.9907578558225508, 0.9852670349907919, 0.9791099000908265, 0.9587813620071685, 0.954337899543379]
    # data[REAL][L2][RE] =[0.9640287769784173, 0.9622302158273381, 0.9694244604316546, 0.9622302158273381, 0.939748201438849]
    # data[REAL][L2][F1] =[0.9772105742935279, 0.9736123748862603, 0.9742431089019431, 0.9605026929982048, 0.9469868599909379]




    #DREBIN CLASSIFICADORES
    labels = [RF, SVM, KNN, DT, AB]
    labels = [RF, KNN, DT]


    data[SYNT][RF] = {}
    data[SYNT][RF][AC] =  [0.9251994680851063, 0.9441489361702128, 0.9082446808510638, 0.8932845744680851, 0.9365026595744681]
    data[SYNT][RF][PR] = [0.9743315508021391, 0.9710578842315369, 0.932712215320911, 0.9048106448311156, 0.9675126903553299]
    data[SYNT][RF][RE] = [0.8192446043165468, 0.875, 0.810251798561151, 0.7949640287769785, 0.8570143884892086]
    data[SYNT][RF][F1] = [0.8900830483634588, 0.9205298013245033, 0.8671799807507219, 0.8463379607467688, 0.9089175011921792]

    data[REAL][RF] = {}
    data[REAL][RF][AC] =  [0.9893617021276596, 0.9817154255319149, 0.9773936170212766, 0.9777260638297872, 0.965093085106383]
    data[REAL][RF][PR] =  [0.9873646209386282, 0.9870967741935484, 0.9736842105263158, 0.963620230700976, 0.9606587374199451]
    data[REAL][RF][RE] =[0.9838129496402878, 0.9631294964028777, 0.9649280575539568, 0.9766187050359713, 0.9442446043165468]
    data[REAL][RF][F1] =[0.9855855855855856, 0.9749658625398271, 0.969286359530262, 0.9700759267530147, 0.9523809523809523]

    data[SYNT][SVM] = {}
    data[SYNT][SVM][AC] = [0.9072164948453608, 0.9281676089125375, 0.8949118722979714, 0.8829398071167276, 0.9221815763219156]
    data[SYNT][SVM][PR] =  [0.9706214689265537, 0.9647302904564315, 0.9373626373626374, 0.9068522483940042, 0.9461382113821138]
    data[SYNT][SVM][RE] = [0.7724820143884892, 0.8363309352517986, 0.7670863309352518, 0.7616906474820144, 0.8372302158273381]
    data[SYNT][SVM][F1] = [0.8602904356534803, 0.8959537572254336, 0.8437190900098911, 0.8279569892473119, 0.8883587786259542]

    data[REAL][SVM] = {}
    data[REAL][SVM][AC] =[0.9906883937479215, 0.984369803791154, 0.9827070169604257, 0.9750581975390755, 0.9664117060192883]

    data[REAL][SVM][PR] = [0.9972477064220183, 0.9845313921747043, 0.980072463768116, 0.9633601429848079, 0.9525514771709938]
    data[REAL][SVM][RE] =[0.9775179856115108, 0.9730215827338129, 0.9730215827338129, 0.9694244604316546, 0.9568345323741008]
    data[REAL][SVM][F1] = [0.9872842870118074, 0.9787426503844414, 0.9765342960288809, 0.9663827879874497, 0.9546882009869898]

    data[SYNT][KNN] = {}
    data[SYNT][KNN][AC] = [0.9228466910542068, 0.9421350182906552, 0.9038909211839042, 0.8992351180578649, 0.9348187562354506]
    data[SYNT][KNN][PR] = [0.9751619870410367, 0.9746963562753036, 0.9194699286442406, 0.9089989888776542, 0.9561752988047809]
    data[SYNT][KNN][RE] = [0.8120503597122302, 0.8660071942446043, 0.8111510791366906, 0.8084532374100719, 0.8633093525179856]
    data[SYNT][KNN][F1] = [0.886162904808636, 0.9171428571428571, 0.8619206880076445, 0.8557829604950022, 0.9073724007561437]

    data[REAL][KNN] = {}
    data[REAL][KNN][AC] =[0.9916860658463585, 0.9830395743265713, 0.9803791153974061, 0.9767209843698038, 0.9670768207515796]
    data[REAL][KNN][PR] =[0.9954421148587056, 0.9871441689623508, 0.9782016348773842, 0.9618794326241135, 0.9651056014692379]
    data[REAL][KNN][RE] = [0.9820143884892086, 0.966726618705036, 0.9685251798561151, 0.9757194244604317, 0.9451438848920863]
    data[REAL][KNN][F1] =[0.9886826618379357, 0.9768287142208089, 0.973339358337099, 0.9687500000000001, 0.9550204452521581]

    data[SYNT][DT] = {}
    data[SYNT][DT][AC] = [0.9228466910542068, 0.9398071167276355, 0.913867642168274, 0.8939142001995344, 0.9318257399401396]

    data[SYNT][DT][PR] = [0.9721030042918455, 0.965965965965966, 0.9383350462487153, 0.9117341640706127, 0.953953953953954]
    data[SYNT][DT][RE] = [0.814748201438849, 0.8678057553956835, 0.8210431654676259, 0.789568345323741, 0.8570143884892086]
    data[SYNT][DT][F1] = [0.8864970645792564, 0.9142586451918522, 0.875779376498801, 0.8462650602409638, 0.9028896257697773]

    data[REAL][DT] = {}
    data[REAL][DT][AC] =[0.9880279348187563, 0.9817093448619887, 0.9807116727635518, 0.9714000665114733, 0.962753574991686]

    data[REAL][DT][PR] = [0.9926739926739927, 0.9870967741935484, 0.9782214156079855, 0.9638336347197106, 0.9570383912248629]
    data[REAL][DT][RE] =[0.9748201438848921, 0.9631294964028777, 0.9694244604316546, 0.9586330935251799, 0.9415467625899281]
    data[REAL][DT][F1] =[0.983666061705989, 0.9749658625398271, 0.973803071364047, 0.9612263300270514, 0.9492293744333636]

    data[SYNT][AB] = {}
    data[SYNT][AB][AC] =[0.9078816095776522, 0.9291652810109744, 0.9092118390422348, 0.8749584303292318, 0.9208513468573329]
    data[SYNT][AB][PR] =[0.9542981501632208, 0.9600818833162743, 0.9276248725790011, 0.8817427385892116, 0.9343936381709742]
    data[SYNT][AB][RE] =[0.7886690647482014, 0.8435251798561151, 0.8183453237410072, 0.7643884892086331, 0.8453237410071942]
    data[SYNT][AB][F1] =[0.8636139832594781, 0.8980373384394447, 0.8695652173913043, 0.8188824662813102, 0.8876298394711992]


    data[REAL][AB] = {}
    data[REAL][AB][AC] =[0.9860325906218823, 0.9807116727635518, 0.9777186564682407, 0.9757233122713668, 0.96474891918856]
    data[REAL][AB][PR] =[0.992633517495396, 0.9906890130353817, 0.9771689497716894, 0.9744292237442922, 0.9692164179104478]
    data[REAL][AB][RE] =[0.9694244604316546, 0.9568345323741008, 0.9622302158273381, 0.9595323741007195, 0.9343525179856115]
    data[REAL][AB][F1] =[0.9808917197452229, 0.9734675205855444, 0.9696420480289986, 0.9669234254644313, 0.9514652014652015]


    plots.append(("drebin_classificadores.pdf", labels, "Classificador"))
    plots.append(("drebin_topologias.pdf", labels2, "Topologia" ))

    for (output, labels, xlabel) in plots:
        #REAL
        # for x in labels:
        #     print("{} accuracy values : {}".format(x, data[REAL][x][AC]))
        #     print("{} accuracy mean   : {}".format(x, np.mean(data[REAL][x][AC])))
        #     print("{} accuracy error  : {}".format(x, np.std(data[REAL][x][AC])))
            
        accuracy_means = [np.mean(data[REAL][x][AC]) for x in labels ]
        accuracy_error = [np.std(data[REAL][x][AC]) for x in labels ]
        #print("Accuracia media: {}".format(accuracy_means))
        #print("Accuracia erro : {}".format(accuracy_error))

        precision_means = [np.mean(data[REAL][x][PR]) for x in labels]
        precision_error = [np.std(data[REAL][x][PR]) for x in labels]

        recall_means = [np.mean(data[REAL][x][RE]) for x in labels]
        recall_error = [np.std(data[REAL][x][RE]) for x in labels]

        f1_means = [np.mean(data[REAL][x][F1]) for x in labels]
        f1_error = [np.std(data[REAL][x][F1]) for x in labels]

        #SYNTHETIC
        accuracy_means_det = [np.mean(data[SYNT][x][AC]) for x in labels]
        accuracy_error_det = [np.std(data[SYNT][x][AC]) for x in labels]

        precision_means_det = [np.mean(data[SYNT][x][PR]) for x in labels]
        precision_error_det = [np.std(data[SYNT][x][PR]) for x in labels]

        recall_means_det = [np.mean(data[SYNT][x][RE]) for x in labels]
        recall_error_det = [np.std(data[SYNT][x][RE]) for x in labels]

        f1_means_det = [np.mean(data[SYNT][x][F1]) for x in labels]
        f1_error_det = [np.std(data[SYNT][x][F1]) for x in labels]

        figheight =  10
        figwidth = 30
        plot_2bars(output, title, labels, xlabel, [REAL, SYNT], plota_linha, figheight, figwidth,
                   accuracy_means, precision_means, recall_means, f1_means,
                   accuracy_error, precision_error, recall_error, f1_error,
                   accuracy_means_det, precision_means_det, recall_means_det, f1_means_det,
                   accuracy_error_det, precision_error_det, recall_error_det, f1_error_det)

    # MSE = 'MSE'
    # CS = 'Similaridade do Cosseno'
    # KL = 'Divergência do KL'
    # MM = 'Max Mean discrepancia'

    labels = [L1N256, L1N1024, L1N4096]
    data2 = {}
    data2[L1N256] = {}
    data2[L1N256][MSE] = [0.17732248, 0.16596623, 0.1689979, 0.16186108, 0.15066549]
    data2[L1N256][CS] = [0.35226875970120286, 0.22406007187446567, 0.3310548239643089, 0.6468441869913639,
                         0.1693229265325205]
    data2[L1N256][KL] = [0.17732248, 0.16596623, 0.1689979, 0.16186108, 0.15066549]
    data2[L1N256][MM] = [0.17732248, 0.16596623, 0.1689979, 0.16186108, 0.15066549]

    data2[L1N1024] = {}
    data2[L1N1024][MSE] = [0.14920367, 0.1488403, 0.14957966, 0.15070726, 0.14779003]
    data2[L1N1024][CS] = [0.22648073915154376, 0.2424270364796259, 0.3685090793962589, 0.2767424347826243,
                          0.2901378215105085]
    data2[L1N1024][KL] = [0.14920367, 0.1488403, 0.14957966, 0.15070726, 0.14779003]
    data2[L1N1024][MM] = [0.14920367, 0.1488403, 0.14957966, 0.15070726, 0.14779003]

    data2[L1N4096] = {}
    data2[L1N4096][MSE] = [0.14863156, 0.15018602, 0.14900272, 0.15022314, 0.15004368]
    data2[L1N4096][CS] = [0.2960275681392151, 0.20929477453395343, 0.3210937559460936, 0.3052652338374125,
                          0.23859077090683037]
    data2[L1N4096][KL] = [0.14863156, 0.15018602, 0.14900272, 0.15022314, 0.15004368]
    data2[L1N4096][MM] = [0.14863156, 0.15018602, 0.14900272, 0.15022314, 0.15004368]

    MSE_means = [np.mean(data2[x][MSE]) for x in labels]
    MSE_error = [np.std(data2[x][MSE]) for x in labels]

    CS_means = [np.mean(data2[x][CS]) for x in labels]
    CS_error = [np.std(data2[x][CS]) for x in labels]

    KL_means = [np.mean(data2[x][KL]) for x in labels]
    KL_error = [np.std(data2[x][KL]) for x in labels]

    MM_means = [np.mean(data2[x][MM]) for x in labels]
    MM_error = [np.std(data2[x][MM]) for x in labels]

    figheight = 10
    figwidth = 30
    plot_1bar("similaridade.pdf", title, labels, xlabel, [REAL], plota_linha, figheight, figwidth,
                CS_means, MSE_means, KL_means, MM_means,
                CS_error, MSE_error, KL_error, MM_error)
 
if __name__ == '__main__':
    sys.exit(main())


