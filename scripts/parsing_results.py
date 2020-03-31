#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:00:44 2020

INthis script I am going to parse the txt results script to make tables with the False positive Rate

@author: labuser
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_restults(files, path, measure, dic):
    '''
    Will parse all the files andcrceate the corresponding heatmaps with 
    the error measure
    '''

    for file in files:
        
        contamination = file.split('_')[-3]
        
        training_dic = {}
        testing_dic = {}
        
        
        
        with open(path+file, 'r') as file_in:
            for line in file_in:
                line = line.rstrip().split(' ')
                if line[0] == 'Coverage':
                    if line[1] in training_dic:
                        training_dic[line[1]] += [float(line[dic[measure][0]])]
                        testing_dic[line[1]] += [float(line[dic[measure][1]])]
                    else:
                        training_dic[line[1]] = [float(line[dic[measure][0]])]
                        testing_dic[line[1]] =  [float(line[dic[measure][1]])]
        
        training_df = pd.DataFrame(training_dic)
        testing_df = pd.DataFrame(testing_dic)
        
        
        maximun_val_train = max(training_df.max())
        minimun_val_train = min(training_df.min())
        
        maximun_val_test = max(testing_df.max())
        minimun_val_test = min(testing_df.min())
        
        maximun_total = max(maximun_val_train, maximun_val_test)
        minimun_total = min(minimun_val_train, minimun_val_test)
            
        
        plt.figure(figsize=(13,9))
        
        plot = sns.heatmap(training_df, 
                           annot=True, 
                           cmap='Blues', 
                           yticklabels = stoichiometries,
                           vmin=minimun_total, 
                           vmax=maximun_total)
        
        plot.xaxis.tick_top() # x axis on top
        plt.title('Trainning '+measure+' '+ 'contamination '+ str(contamination))
        path_out = '/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(contamination)+'_'+measure+'_train.pdf'
        plt.savefig(path_out,
                    format='pdf',
                    dpi=1200,
                    bbox_inches='tight', pad_inches=0)
        plt.show()
        
        ######
        
        plt.figure(figsize=(13,9))
        
        plot = sns.heatmap(testing_df, 
                           annot=True, 
                           cmap='Blues', 
                           yticklabels = stoichiometries,
                           vmin=minimun_total, 
                           vmax=maximun_total)
        
        plot.xaxis.tick_top() # x axis on top
        plt.title('Testing '+measure+' '+ 'contamination '+ str(contamination))
        path_out = '/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(contamination)+'_'+measure+'_test.pdf'
        plt.savefig(path_out,
                    format='pdf',
                    dpi=1200,
                    bbox_inches='tight', pad_inches=0)
        plt.show()
        
    return True


path  = '/media/labuser/Data/nanopore/UNION/results/throughput_results/'

files = os.listdir(path)
files = [i for i in files if i[-3:] == 'txt']

stoichiometries = [100, 75, 50, 25, 10]

dic = {'FPR' : [13, 17], 'FNR':[21, 25], 'distance_toi': [29, 33]}

files.pop(1)
files.pop(1)
files.pop(1)
files.pop(2)

parse_restults(files, path, 'FPR', dic)
parse_restults(files, path, 'FNR', dic)
parse_restults(files, path, 'distance_toi', dic)






































