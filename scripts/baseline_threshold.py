#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:33:08 2020

@author: labuser

This sript will test the baseline of one-call methods using pUC19 modified and no modified reads
"""

from sklearn import svm
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import umap
import seaborn as sns
import math

def calculate_accuracies(model, test_data, fitter, training_n, testing_n, f, dic):
        
    #test 
    #fit the model with the training no mod data
    
    model.fit(fitter.embedding_[:training_n])
    
    # predict and accuraacy for the training the training
    prediction = model.predict(fitter.embedding_)
    n_error_train = prediction[:training_n][prediction[:training_n] == -1].size
    n_error_outliers = prediction[training_n:][prediction[training_n:] == 1].size
    f.write('Ac training'+str((len(prediction[:training_n])-n_error_train)/len(prediction[:training_n]))+'\n')
    f.write('Ac mod testing'+str((len(prediction[training_n:])-n_error_outliers)/len(prediction[training_n:]))+'\n')
    
    prediction = model.predict(test_data)
    n_error_test = prediction[:testing_n][prediction[:testing_n] == -1].size
    n_error_outliers = prediction[testing_n:][prediction[testing_n:] == 1].size
    f.write('Ac testing no mod'+str((len(prediction[:testing_n])-n_error_test)/len(prediction[:testing_n]))+'\n')
    f.write('Ac testing mod'+str((len(prediction[testing_n:])-n_error_outliers)/len(prediction[testing_n:]))+'\n')
    f.write('\n')
    if 'testing_no_mod' not in dic:
        dic['testing_no_mod'] = [(len(prediction[:testing_n])-n_error_test)/len(prediction[:testing_n])]
    else:
        dic['testing_no_mod'] = dic['testing_no_mod'] + [(len(prediction[:testing_n])-n_error_test)/len(prediction[:testing_n])]
    
    if 'testing_mod' not in dic:
        dic['testing_mod'] = [(len(prediction[testing_n:])-n_error_outliers)/len(prediction[testing_n:])]
    else:
        dic['testing_mod'] =  dic['testing_mod'] + [(len(prediction[testing_n:])-n_error_outliers)/len(prediction[testing_n:])]
    
    return True



def calculate_accuracies_raw(model, test_data, train_data, training_n, testing_n, f):
        
    #test 
    #fit the model with the training no mod data
    
    model.fit(train_data[:training_n])
    
    # predict and accuraacy for the training the training
    prediction = model.predict(train_data)
    
    n_error_train = prediction[:training_n][prediction[:training_n] == -1].size
    n_error_outliers = prediction[:training_n][prediction[training_n:] == 1].size
    f.write('Ac raw training'+str((training_n-n_error_train)/training_n)+'\n')
    f.write('Ac raw mod testing'+str((training_n-n_error_outliers)/training_n)+'\n')
    
    prediction = model.predict(test_data)
    n_error_test = prediction[:testing_n][prediction[:testing_n] == -1].size
    n_error_outliers = prediction[testing_n:][prediction[testing_n:] == 1].size
    f.write('Ac raw testing no mod'+str((testing_n-n_error_test)/testing_n)+'\n')
    f.write('Ac raw testing mod'+str((testing_n-n_error_outliers)/testing_n)+'\n')
    f.write('\n')
    return True


def euclidean_distances(signal_data):
    '''
    remove outliers to have a better fit ti the training data
    '''
    df = pd.DataFrame()
    for i in range(len(signal_data)):
        temp_column = []
        for j in range(len(signal_data)):
            temp_column.append(math.sqrt(np.sum((signal_data[i] - signal_data[j]) ** 2)))
        df[i] = temp_column
    return df
    
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
    

path = '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/1000/'


motif = ['/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_CCTGGCCTT_700.npy',
         '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_CCTGGCGTT_700.npy',
         '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_CCTGGTATCT_700.npy',
         path+'motif_CCAGGAACC_1000.npy' ,
         path+'motif_CCAGGCGTTT_1000.npy' ,
         path+'motif_CCAGGGTTT_1000.npy' ,
         path+'motif_CCTGGAAGC_1000.npy',
         path+'motif_CCTGGGGTG_1000.npy'
        ]


motif_mod = [
            '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_mod_CCTGGCCTT_700.npy',
            '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_mod_CCTGGCGTT_700.npy',
            '/media/labuser/Data/nanopore/pUC19/processed/numpy/tombo/n_prepro/5-mers/motif_mod_CCTGGTATCT_700.npy',
             path+'motif_mod_CCAGGAACC_1000.npy',
             path+'motif_mod_CCAGGCGTTT_1000.npy',
             path+'motif_mod_CCAGGGTTT_1000.npy',
             path+'motif_mod_CCTGGAAGC_1000.npy',
             path+'motif_mod_CCTGGGGTG_1000.npy'
            ]

number_training = [600,600,600,900,900,900,900,900]


LOF = {}
iso = {}
envelop = {}

file_out = '/media/labuser/Data/nanopore/UNION/results/baseline_threshold'
f = open(file_out, "w")


for i in range(len(motif)):
    
    f.write(motif[i]+'\n')
    no_mod = np.load(motif[i])
    mod = np.load(motif_mod[i])
    number = number_training[i]
    
    # split no mod into traininig data and testing one
    # modified does not have to be partition
    no_mod_train = no_mod[:number] 
    mod_train = mod[:number]
    
    x = np.concatenate((no_mod_train,mod_train))
    
    ## extract good features
    fitter = umap.UMAP(n_neighbors =50,
                       set_op_mix_ratio=0.1,
                       n_epochs = 1000,
                       learning_rate=0.5,
                       min_dist=1,
                       random_state=42,
                       n_components=4,
                       metric = 'chebyshev').fit(x.reshape((len(x)), 50),
                                                )
    
    test_data = fitter.transform(np.concatenate((no_mod[number:], mod[number:])))
    
    model_LocalOutlierFactor = LocalOutlierFactor(
                n_neighbors=20, contamination=0.2, novelty=True, leaf_size=10)
    
    
    #f.write('LOF'+'\n')
    #calculate_accuracies(model_LocalOutlierFactor, test_data, fitter, number, 100, f, LOF)
    
    model_LocalOutlierFactor.fit(fitter.embedding_[:number])
        
    # predict and accuraacy for the training the training
    prediction = model_LocalOutlierFactor.predict(test_data)
    labels = np.concatenate((np.ones(100), np.repeat(-1, 100)))
    f.write('testing accuracy without threshold '+str(accuracy(labels, prediction))+'\n')
    
    # selected extra-outliers
    decision = model_LocalOutlierFactor.decision_function(test_data)
    index = []
    for i in range(len(decision)):
        if decision[i] < 0.02 and decision[i] > -0.1:
            index.append(i)
    
    #get rid of non-confident ones
    mod_training_filtered = np.delete(test_data, index, axis = 0)
    f.write('Number of signals left '+str(len(mod_training_filtered))+'\n')
    labels = np.concatenate((np.ones(100), np.repeat(-1, 100)))
    labels = np.delete(labels, index, axis = 0)
    prediction = model_LocalOutlierFactor.predict(mod_training_filtered)
    f.write('testing accuracy with threshold '+str(accuracy(labels, prediction))+'\n')
    
    
f.close()
