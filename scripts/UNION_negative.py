#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:58:18 2020

@author: labuser
In this script I am going to try our method using two motif that are not modified, as if the WT has no modification
Selecting contamination 0.1 which is the one that gave me the best results
"""

from sklearn import svm
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import umap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd


def accuracy(labels, predictions):
    '''
    calculate accuracy
    '''
    if labels.shape != predictions.shape:
        print('labels and predictions does not have same dimentions')
        return False
    
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct +=1
    
    return correct/len(labels)



path = '/media/labuser/Data/nanopore/pUC19_nanopolish/numpy/'


motif = [
         path + 'no_mod_negative_np1019.npy',
         path +'no_mod_negative_np1170.npy',
         path +'no_mod_negative_np190.npy',
         path +'no_mod_negative_np1932.npy',
         path +'no_mod_negative_np2266.npy',
         path +'no_mod_negative_np271.npy',
         path +'no_mod_negative_np653.npy',
         path +'no_mod_negative_np727.npy',
         path +'no_mod_negative_np86.npy',
         path +'no_mod_negative_np976.npy'
        ]


motif_mod =[
           path +'mod_negative_np1019.npy',
           path +'mod_negative_np1170.npy',
           path +'mod_negative_np190.npy',
           path +'mod_negative_np1932.npy',
           path +'mod_negative_np2266.npy',
           path +'mod_negative_np271.npy',
           path +'mod_negative_np653.npy',
           path +'mod_negative_np727.npy',
           path +'mod_negative_np86.npy',
           path +'mod_negative_np976.npy'
          ]


# Parameters    
contaminations = [0.1]

for Contamination in contaminations:
    
    file_out = '/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_negative.txt'
    f = open(file_out, "w")
    
    coverages = [900, 700, 500, 250, 100, 50, 25, 10]
    
    # Store values to make the plots
    training_heat = pd.DataFrame()
    testing_heat = pd.DataFrame()
    
    
    for coverage in coverages:
        
        median_test_acc = []
        median_false_posit_test = []
        median_false_posit_train = []
        median_train_acc = [] 
        f.write('\n')   
        
        for i in range(len(motif)):
        
            #f.write(motif[i]+'\n')
            no_mod = np.load(motif[i])
            mod = np.load(motif_mod[i])
            
            # split no mod into traininig data and testing one
            # modified does not have to be partition
            no_mod_train = no_mod[:coverage] 
            mod_train = mod[:coverage]
            
            x = np.concatenate((no_mod_train, mod_train))
            
            ## extract good features
            fitter = umap.UMAP().fit(x.reshape((len(x)), 60))
                            
            '''
            sns.scatterplot(x=fitter.embedding_[:1350,0], 
                            y=fitter.embedding_[:1350,1], 
                            color="blue", 
                            label="No modified"
                            )
            
            sns.scatterplot(x=fitter.embedding_[1350:,0], 
                            y=fitter.embedding_[1350:,1], 
                            color="red", 
                            label="modified")
            
            plt.show()
            '''
            
            test_data = fitter.transform(np.concatenate((no_mod[-100:], mod[-100:])))
            
            model_EllipticEnvelope =  EllipticEnvelope(contamination=Contamination,
                                                      support_fraction=1)
            
            model_EllipticEnvelope.fit(fitter.embedding_[:len(no_mod_train)])
            
            # calculate training accuracy
            labels = np.concatenate((np.ones(len(no_mod_train)+number_no_mod + number_mod)))
            prediction = model_EllipticEnvelope.predict(fitter.embedding_)
            median_false_posit_train.append(len(prediction[:len(no_mod_train)+number_no_mod][prediction[:len(no_mod_train)+number_no_mod] == -1])/len(prediction[:len(no_mod_train)+number_no_mod]))
            median_train_acc.append(accuracy(labels, prediction))
            
            #get rid of non-confident ones
            labels = np.concatenate((np.ones(100), np.repeat(-1, 100)))
            prediction = model_EllipticEnvelope.predict(test_data)
            median_false_posit_test.append(len(prediction[:100][prediction[:100] == -1])/100)
            #f.write('testing accuracy with threshold '+str(accuracy(labels, prediction))+'\n')
            median_test_acc.append(accuracy(labels, prediction))
            
            train.append(np.mean(median_train_acc))
            test.append(np.mean(median_test_acc))
             
            f.write('Coverage '+str(coverage)+' Stoichiometry '+str(stoi))
            f.write(' Training acc '+str(np.mean(median_train_acc)))
            f.write(' Testing acc '+str(np.mean(median_test_acc)))
            f.write(' False positives train'+str(np.mean(median_false_posit_test)))
            f.write(' False positives testing'+str(np.mean(median_false_posit_train)))
            f.write('\n')
    
        training_heat[coverage] = train
        testing_heat[coverage] = test
        
    f.close()
    
    
    plt.figure(figsize=(13,9))
    
    plot = sns.heatmap(training_heat, annot=True, cmap='YlOrRd', yticklabels = stoichiometries)
    
    plot.xaxis.tick_top() # x axis on top
    plt.title('Trainning dataset Coverage vs stoichiometry')
    plt.savefig('/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_cover_stoi_train.pdf',
                format='pdf',
                dpi=1200,
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
    ######
    
    plt.figure(figsize=(13,9))
    
    plot = sns.heatmap(testing_heat, annot=True, cmap='YlOrRd',yticklabels = stoichiometries)
    
    plot.xaxis.tick_top() # x axis on top
    plt.title('Testing dataset Coverage vs stoichiometry')
    plt.savefig('/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_cover_stoi_test.pdf',
                format='pdf',
                dpi=1200,
                bbox_inches='tight', pad_inches=0)
    plt.show()
