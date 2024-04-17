# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:51:56 2023

@author: Ikaro
"""
import pandas as pd
import json
import numpy as np
from process_VIA_files import get_length, get_manual_classification
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from matplotlib import pyplot as plt

#### COISAS PRA FAZER #####
# 1 - pegar o nome do trial (trial_name)
# 2 - mudar o beginning e end de acordo com o trial

# Primeiro roda o process_VIA_files e pega o nome dos trials 1 por 1
## Main function
# CSV filename
filename = "E:\\AnalisesBia\\VideoAnnotation\\barnes.csv"
# Video_folder
video_folder = "E:\\AnalisesBia\\BarnesMaze\\"
output_dict = get_manual_classification(filename, video_folder)

# Uma vez que rodou o VIA altera o nome dos trials 1 a 1 aqui em baixo
trial_name = 'C38_1_G3_D2_T3'
# Automatic classification path
automatic_filename = "E:\\AnalisesBia\\BarnesMaze\\"+trial_name+".txt"
manual_classification = output_dict[trial_name+".mp4"]

print('Trial: '+trial_name)

# Compare the two exploration temporal series

# First step: get the original deeplabcut file (make SURE the time cut from the beginning and the end is replicated on the manual classification)
df = pd.read_hdf("E:\AnalisesBia\BarnesMaze\Final_results.h5") # NÃO PRECISA MEXER
# 1.1 Get the Beginning value
beginning =81
# 1.2 Get the End value
end = 2557

# Second step: load the automatic classification 
f = open(automatic_filename, )
data = json.load(f)
automatic_classification = np.asarray(data["bp_pos_on_maze_filtered"])

# Third step: trim the manual classification
aux_manual = manual_classification     # Create auxiliar array
# Define the index arrays (for deletion)
delete_array = np.concatenate((np.arange(0,beginning),np.arange(end+1,len(aux_manual))))
# Delete the selected index elements
manual_classification = np.delete(manual_classification, delete_array)

# Forth step: ANNOUCE IF THE AUTOMATIC AND MANUAL ARRAYS DOESN'T HAVE THE SAME LENGTH (ABSURDO!!)
assert len(manual_classification) == len(automatic_classification), "The classisfications have different lengths"
# Transform any -1 classification into 0 (for barnes maze specially)
aux_auto = automatic_classification     # Auxiliar array de lei né
automatic_classification[automatic_classification == -1] = 0

######################### AGORA OS CÁLCULO DE VERDADE SÓ MAGIA BRABA ###################################

# Calcula a matriz de confusão
conf_matrix = confusion_matrix(manual_classification, automatic_classification)

# Calcula precisão, recall, F1-score e suporte para cada classe
precision, recall, fscore, support = precision_recall_fscore_support(manual_classification, automatic_classification)

# Calcula as métricas globais
total_precision = precision.mean()
total_recall = recall.mean()
total_fscore = fscore.mean()

# Extrai TP (Verdadeiros Positivos) e FP (Falsos Positivos)
tp = conf_matrix.diagonal()
fp = conf_matrix.sum(axis=0) - tp
tn = conf_matrix.sum() - (tp + fp + conf_matrix.sum(axis=1) - tp)

# Calcula as taxas
tpr = tp / (tp + conf_matrix.sum(axis=1) - tp)
fpr = fp / (fp + tn)
# Calcula as taxas totais (médias ponderadas)
total_tpr = sum(tp) / sum(tp + conf_matrix.sum(axis=1) - tp)
total_fpr = sum(fp) / sum(fp + tn)


# Exibe os resultados
print("Matriz de Confusão:")
print(conf_matrix)
print("\nPrecisão para cada classe:", precision)
print("Recall para cada classe:", recall)
print("F1-score para cada classe:", fscore)
print("Suporte para cada classe:", support)
print("\nVerdadeiros Positivos para cada classe:", tp)
print("Falsos Positivos para cada classe:", fp)
print("\nTaxa de Verdadeiro Positivo (TPR) para cada classe:", tpr)
print("Taxa de Falso Positivo (FPR) para cada classe:", fpr)

print("\nPrecisão Média Global:", total_precision)
print("Recall Médio Global:", total_recall)
print("F1-score Médio Global:", total_fscore)
print("\nTaxa de Verdadeiro Positivo (TPR) total:", total_tpr)
print("Taxa de Falso Positivo (FPR) total:", total_fpr)


## Plot the classifications
plt.plot(manual_classification, label='Manual')
plt.plot(automatic_classification, label='Auto')
plt.title(trial_name)
plt.legend()
plt.draw()
    