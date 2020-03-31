#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:33:08 2020

@author: labuser

This sript will test the baseline of one-call methods using pUC19 modified and no modified reads
"""

import numpy as np
from sklearn.covariance import EllipticEnvelope
import umap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
         path+'no_mod354_CCAGG_np.npy',
         path+'no_mod545_CCTGG_np.npy',
         path+'no_mod833_CCAGG_np.npy',
         path+'no_mod954_CCAGG_np.npy',
         path+'no_mod967_CCTGG_np.npy',
         path+'no_mod351_CCAGG_r_np.npy',
         path+'no_mod542_CCTGG_r_np.npy',
         path+'no_mod830_CCAGG_r_np.npy',
         path+'no_mod951_CCAGG_r_np.npy',
         path+'no_mod964_CCTGG_r_np.npy'
        ]

motif_stoi = [
         path+'no_mod354_CCAGG_f_2000_np.npy',
         path+'no_mod545_CCTGG_f_2000_np.npy',
         path+'no_mod833_CCAGG_f_2000_np.npy',
         path+'no_mod954_CCAGG_f_2000_np.npy',
         path+'no_mod967_CCTGG_f_2000_np.npy',
         path+'no_mod351_CCAGG_r_2000_np.npy',
         path+'no_mod542_CCTGG_r_2000_np.npy',
         path+'no_mod830_CCAGG_r_2000_np.npy',
         path+'no_mod951_CCAGG_r_2000_np.npy',
         path+'no_mod964_CCTGG_r_2000_np.npy'
        ]


motif_mod = [
             path+'mod354_CCAGG_np.npy',
             path+'mod545_CCTGG_np.npy',
             path+'mod833_CCAGG_np.npy',
             path+'mod954_CCAGG_np.npy',
             path+'mod967_CCTGG_np.npy',
             path+'mod351_CCAGG_r_np.npy',
             path+'mod542_CCTGG_r_np.npy',
             path+'mod830_CCAGG_r_np.npy',
             path+'mod951_CCAGG_r_np.npy',
             path+'mod964_CCTGG_r_np.npy'
        ]


# Parameteres
contaminations = [0.05, 0.3, 0.2, 0.15, 0.1]

for Contamination in contaminations:
    
    file_out = '/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_cover_stoi.txt'
    f = open(file_out, "w")
    
    coverages = [900, 700, 500, 250, 100, 50, 25, 10]
    stoichiometries = [100, 75, 50, 25, 10]
    
    # Store values to make the plots
    training_heat = pd.DataFrame()
    testing_heat = pd.DataFrame()
    
    for coverage in coverages:
        
        train = []
        test = []
        
        for stoi in stoichiometries:
            
            median_test_acc = []
            median_train_acc = [] 
            median_false_posit_test = []
            median_false_posit_train = []
            median_false_negative_train = []
            median_false_negative_test = []
            median_distance_stoi_train = []
            median_distance_stoi_test = []
            
            f.write('\n')   
            
            for i in range(len(motif)):
            
                #f.write(motif[i]+'\n')
                no_mod = np.load(motif[i])
                mod = np.load(motif_mod[i])
                no_mod_soti = np.load(motif_stoi[i])
                
                # split no mod into traininig data and testing one
                # modified does not have to be partition
                no_mod_train = no_mod[:coverage] 
                
                number_mod = int(coverage * (stoi/100))
                if number_mod == coverage:
                    number_no_mod = 0
                    mod_train = mod[:number_mod]
                else:
                    number_no_mod = coverage - number_mod
                    mod_train = np.concatenate((no_mod_soti[-number_no_mod:], mod[:number_mod]))
                
                x = np.concatenate((no_mod_train, mod_train))
                
                # Choose number of n_neighbors
                n_neighbors = int(len(x)**(1/2))
                if n_neighbors > 15:
                    n_neighbors == 15
                
                ## extract good features
                fitter = umap.UMAP(n_neighbors=n_neighbors).fit(x.reshape((len(x)), 60))
                                
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
                labels = np.concatenate((np.ones(len(no_mod_train)+number_no_mod), np.repeat(-1, number_mod)))
                prediction = model_EllipticEnvelope.predict(fitter.embedding_)
                median_train_acc.append(accuracy(labels, prediction))
                prediction_no_mod = prediction[:len(no_mod_train)+number_no_mod]
                prediction_mod =  prediction[len(no_mod_train)+number_no_mod:]
                median_false_posit_train.append(len(prediction_no_mod[prediction_no_mod == -1])/len(prediction_no_mod))
                median_false_negative_train.append(len(prediction_mod[prediction_mod == 1])/len(prediction_mod))
                stoichiometry_train = prediction[len(no_mod_train):]
                stoichiometry_train = len(stoichiometry_train[stoichiometry_train == -1])/len(stoichiometry_train)
                median_distance_stoi_train.append(abs((stoi - stoichiometry_train*100)))
                
                
                #get rid of non-confident ones
                labels = np.concatenate((np.ones(100), np.repeat(-1, 100)))
                prediction = model_EllipticEnvelope.predict(test_data)
                median_false_posit_test.append(len(prediction[:100][prediction[:100] == -1])/100)
                median_false_negative_test.append(len(prediction[100:][prediction[100:] == 1])/100)
                #f.write('testing accuracy with threshold '+str(accuracy(labels, prediction))+'\n')
                median_test_acc.append(accuracy(labels, prediction))
                stoichiometry_test = len(prediction[100:][prediction[100:] == -1])/len(prediction[100:])
                median_distance_stoi_test.append(abs(100 - stoichiometry_test*100))
                
                
            train.append(np.mean(median_train_acc))
            test.append(np.mean(median_test_acc))
             
            f.write('Coverage '+str(coverage)+' Stoichiometry '+str(stoi))
            f.write(' Training acc '+str(np.mean(median_train_acc)))
            f.write(' Testing acc '+str(np.mean(median_test_acc)))
            f.write(' False positives train '+str(np.mean(median_false_posit_train)))
            f.write(' False positives testing '+str(np.mean(median_false_posit_test)))
            f.write(' False negative train '+str(np.mean(median_false_negative_train)))
            f.write(' False negative testing '+str(np.mean(median_false_negative_test)))
            f.write(' Distance stoichio. train '+str(np.mean(median_distance_stoi_train)))
            f.write(' Distance stoichio. test '+str(np.mean(median_distance_stoi_test)))
            f.write('\n')
    
        training_heat[coverage] = train
        testing_heat[coverage] = test
        
    f.close()
    
    #fix the heatmap scale
    maximun_val_train = max(training_heat.max())
    minimun_val_train = min(training_heat.min())
    
    maximun_val_test = max(testing_heat.max())
    minimun_val_test = min(testing_heat.min())
    
    maximun_total = max(maximun_val_train, maximun_val_test)
    minimun_total = min(minimun_val_train, minimun_val_test)
    
    plt.figure(figsize=(13,9))
    
    plot = sns.heatmap(training_heat, 
                       annot=True, 
                       cmap='YlOrRd', 
                       yticklabels = stoichiometries, 
                       vmin=minimun_total, 
                       vmax=maximun_total)
    
    plot.xaxis.tick_top() # x axis on top
    plt.title('Trainning dataset Coverage vs stoichiometry')
    plt.savefig('/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_cover_stoi_train.pdf',
                format='pdf',
                dpi=1200,
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
    ######
    
    plt.figure(figsize=(13,9))
    
    plot = sns.heatmap(testing_heat, 
                       annot=True, 
                       cmap='YlOrRd', 
                       yticklabels = stoichiometries,
                       vmin=minimun_total, 
                       vmax=maximun_total
                       )
    
    plot.xaxis.tick_top() # x axis on top
    plt.title('Testing dataset Coverage vs stoichiometry')
    plt.savefig('/media/labuser/Data/nanopore/UNION/results/throughput_results/UNION_pUC19_nanopolish_contamination_'+str(Contamination)+'_cover_stoi_test.pdf',
                format='pdf',
                dpi=1200,
                bbox_inches='tight', pad_inches=0)
    plt.show()
