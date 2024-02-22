import json
import os
import sys
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
def create_argparse(): 
           parser=argparse.ArgumentParser()
           parser.add_argument('-f','--feature_dataset',type=str,default=False,help='Csv file contaning the features with p_value score (csv)')
           parser.add_argument('-num','--number_of_features',type=int,required=True,help='Number of features to be select from the feature file')
           parser.add_argument('-d','--data_dataset',type=str,default={},help=' Matrix of data without the title in its collumns  (npy)')
           parser.add_argument('-nf','--name_features',type=str,default={},help='Csv file contaning the title of collumns')
           parser.add_argument('-o','--output_dir',type=str,default=f'out/',help='directory in which to save the resulting file')
           parser.add_argument('-n','--name_file',type=str,default=f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',help='name of the file to be saved')
           parser.add_argument('-l','--labels',type=str,default='processed/mh100_vt-labels.csv',help='labels to be utilized in the feature selection')
           return  parser.parse_args()

def selection(features,number_of_features):
    chi2=pd.read_csv(features)
    chi2_sorted=chi2.sort_values(by='stats',ascending=False).dropna()
    chi2_features=chi2_sorted[chi2_sorted['p_values']<0.05].head(number_of_features)
    return chi2_features

def treatment(initial_arguments):
    if initial_arguments.name_features !=None:
       features_names=pd.read_csv(initial_arguments.name_features,index_col=0)
    else:
       features_names=None
    if initial_arguments.data_dataset != None:
       dataset=np.load(initial_arguments.data_dataset)
    else:
        dataset=None
    #if (dataset or features_names == None):
        # if initial_arguments==False
         #    print("Error must provide a dataset and collum name or a file with features sorted")
        # else: 
             # features=selection(initial_arguments.feature_dataset,initial_arguments_number_of_features)  
    #else:
       # dataset_file=pd.DataFrame(dataset,columns=features_names.features.values)
       #vt_labels=pd.read_csv(initial_arguments.labels)
      # chi2_stats,p_values=chi2(dataset_file,initi,vt_labels['4-class'])
    #features=features.dropna()

    features=selection(initial_arguments.feature_dataset,initial_arguments.number_of_features)
    dataset_file=pd.DataFrame(dataset,columns=features_names.features.values)
    labels=pd.read_csv("processed/mh100_labels.csv")
    features=features.set_index('names')
    dataset_file_loaded=dataset_file[dataset_file.columns.intersection(features.index.values)]
    dataset_file_loaded['class']=labels.loc[:,'CLASS']
    
    #dataset_file_loaded.to_csv(initial_arguments.output_dir+initial_arguments.name_file)

if __name__ == "__main__":
   arguments=create_argparse()
  
    #if arguments.verbosity == logging.DEBUG:
        # mostra mais detalhes
        #logging_format = '%(asctime)s\t***\t%(levelname)s {%(module)s} [%(funcName)s] %(message)s' 
   Path(arguments.output_dir).mkdir(parents=True, exist_ok=True)
    #logging_filename = os.path.join(arguments.output_dir, LOGGING_FILE_NAME)
   treatment(arguments)
