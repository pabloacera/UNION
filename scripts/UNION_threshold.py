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
    remove outliers to have a better fit to the training data
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

number_training = [900, 900, 900, 900, 900, 900, 900, 900, 900, 900]


LOF = {}
iso = {}
envelop = {}

file_out = '/media/labuser/Data/nanopore/UNION/results/UNOIN_pUC19_nanopolish_thershold_contamination_0.05.txt'
f = open(file_out, "w")
    
thresholds = [0, -1.5, -2, -2.5, -3, -3.5, -4]

for threshold in thresholds:
    
    median_number = []
    median_acc = []
    median_false_posit = []
    
    for i in range(len(motif)):
    
        #f.write(motif[i]+'\n')
        no_mod = np.load(motif[i])
        mod = np.load(motif_mod[i])
        number = number_training[i]
        
        # split no mod into traininig data and testing one
        # modified does not have to be partition
        no_mod_train = no_mod[:number] 
        mod_train = mod[:number]
        
        x = np.concatenate((no_mod_train,mod_train))
        
        ## extract good features
        fitter = umap.UMAP().fit(x.reshape((len(x)), 60))
        
        test_data = fitter.transform(np.concatenate((no_mod[number:], mod[number:])))
        
        model_EllipticEnvelope =  EllipticEnvelope(contamination=0.05,
                                                  support_fraction=1)
        
        model_EllipticEnvelope.fit(fitter.embedding_[:number])
        
        # selected extra-outliers
        decision = model_EllipticEnvelope.decision_function(test_data)
        index = []
        for i in range(len(decision)):
            if decision[i] < 0.00 and decision[i] > threshold:
                index.append(i)
        
        #get rid of non-confident ones
        mod_training_filtered = np.delete(test_data, index, axis = 0)
        #f.write('Number of signals left '+str(len(mod_training_filtered))+'\n')
        median_number.append(len(mod_training_filtered))
        labels = np.concatenate((np.ones(100), np.repeat(-1, 100)))
        labels = np.delete(labels, index, axis = 0)
        prediction = model_EllipticEnvelope.predict(mod_training_filtered)
        median_false_posit.append(len(prediction[:100][prediction[:100] == -1]))
        #f.write('testing accuracy with threshold '+str(accuracy(labels, prediction))+'\n')
        median_acc.append(accuracy(labels, prediction))
        
    
    f.write('Decision boundary '+str(threshold))
    f.write(' Mean number '+str(np.mean(median_number)))
    f.write(' Mean acc '+str(np.mean(median_acc)))
    f.write(' Mean False positives '+str(np.mean(median_false_posit)))
    f.write('\n')
    
f.close()
