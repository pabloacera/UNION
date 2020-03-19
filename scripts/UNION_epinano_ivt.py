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


def calculate_accuracies(model, test_data, fitter, training_n, testing_n, f, dic):
        
    #test 
    #fit the model with the training no mod data
    
    model.fit(fitter.embedding_[:training_n])
    
    # predict and accuraacy for the training the training
    prediction = model.predict(fitter.embedding_)
    n_error_train = prediction[:training_n][prediction[:training_n] == -1].size
    n_error_outliers = prediction[training_n:][prediction[training_n:] == 1].size
    f.write('Ac training'+str((training_n-n_error_train)/training_n)+'\n')
    f.write('Ac mod testing'+str((training_n-n_error_outliers)/training_n)+'\n')
    
    prediction = model.predict(test_data)
    n_error_test = prediction[:testing_n][prediction[:testing_n] == -1].size
    n_error_outliers = prediction[testing_n:][prediction[testing_n:] == 1].size
    f.write('Ac testing no mod'+str((testing_n-n_error_test)/testing_n)+'\n')
    f.write('Ac testing mod'+str((testing_n-n_error_outliers)/testing_n)+'\n')
    f.write('\n')
    if 'testing_no_mod' not in dic:
        dic['testing_no_mod'] = [(len(prediction[:testing_n])-n_error_test)/testing_n]
    else:
        dic['testing_no_mod'] = dic['testing_no_mod'] + [(len(prediction[:testing_n])-n_error_test)/testing_n]
    
    if 'testing_mod' not in dic:
        dic['testing_mod'] = [(len(prediction[testing_n:])-n_error_outliers)/testing_n]
    else:
        dic['testing_mod'] =  dic['testing_mod'] + [(len(prediction[testing_n:])-n_error_outliers)/testing_n]

    return True


def calculate_accuracies_raw(model, test_data, train_data, training_n, testing_n, f):
        
    #test 
    #fit the model with the training no mod data
    
    model.fit(train_data[:training_n])
    
    # predict and accuraacy for the training the training
    prediction = model.predict(train_data)
    
    n_error_train = prediction[:training_n][prediction[:training_n] == -1].size
    n_error_outliers = prediction[training_n:][prediction[training_n:] == 1].size
    f.write('Ac raw training'+str((training_n-n_error_train)/training_n)+'\n')
    f.write('Ac raw mod testing'+str((training_n-n_error_outliers)/training_n)+'\n')
    
    prediction = model.predict(test_data)
    n_error_test = prediction[:testing_n][prediction[:testing_n] == -1].size
    n_error_outliers = prediction[testing_n:][prediction[testing_n:] == 1].size
    f.write('Ac raw testing no mod'+str((testing_n-n_error_test)/testing_n)+'\n')
    f.write('Ac raw testing mod'+str((testing_n-n_error_outliers)/testing_n)+'\n')
    f.write('\n')
    return True


# Epinano IVT motifs

path_no_mod ='/media/labuser/Data/nanopore/Epinanot_IVT/no_mod/numpy2/'

motif = [path_no_mod+'pos_1353_cc6m_2244_t7_ecorv.npy',
         path_no_mod+'pos_1418_cc6m_2595_t7_ecorv.npy',
         path_no_mod+'pos_1667_cc6m_2459_t7_ecorv.npy',
         path_no_mod+'pos_300_cc6m_2459_t7_ecorv.npy',
         path_no_mod+'pos_33_cc6m_2244_t7_ecorv.npy',
         path_no_mod+'pos_405_cc6m_2595_t7_ecorv.npy',
         path_no_mod+'pos_463_cc6m_2459_t7_ecorv.npy',
         path_no_mod+'pos_754_cc6m_2459_t7_ecorv.npy',
         path_no_mod+'pos_845_cc6m_2595_t7_ecorv.npy',
         path_no_mod+'pos_106_cc6m_2459_t7_ecorv.npy'
        ]

 
path_mod = '/media/labuser/Data/nanopore/Epinanot_IVT/mod/numpy2/'

motif_mod = [path_mod+'pos_1353_cc6m_2244_t7_ecorv.npy',
             path_mod+'pos_1418_cc6m_2595_t7_ecorv.npy',
             path_mod+'pos_1667_cc6m_2459_t7_ecorv.npy',
             path_mod+'pos_300_cc6m_2459_t7_ecorv.npy',
             path_mod+'pos_33_cc6m_2244_t7_ecorv.npy',
             path_mod+'pos_405_cc6m_2595_t7_ecorv.npy',
             path_mod+'pos_463_cc6m_2459_t7_ecorv.npy',
             path_mod+'pos_754_cc6m_2459_t7_ecorv.npy',
             path_mod+'pos_845_cc6m_2595_t7_ecorv.npy',
             path_mod+'pos_106_cc6m_2459_t7_ecorv.npy'
            ]


#number_training = [900, 900, 900, 900, 900]

LOF = {}
iso = {}
envelop = {}

file_out = '/media/labuser/Data/nanopore/UNION/results/UNION_epinano_500_10motifs.txt'
f = open(file_out, "w")

for i in range(len(motif)):
    
    no_mod = np.load(motif[i])
    mod = np.load(motif_mod[i])
    number = 500
                            
    # split no mod into traininig data and testing one
    # modified does not have to be partition
    no_mod_train = no_mod[:500] 
    mod_train = mod[:500]
    
    x = np.concatenate((no_mod_train, mod_train))
        
    '''
    X_embedded = TSNE(n_components=2)
    X_embedded =  X_embedded.fit_transform(x)
    
    sns.scatterplot(x=X_embedded[:203,0], 
                    y=X_embedded[:203,1], 
                    color="blue", 
                    label="No modified"
                    )
    
    sns.scatterplot(x=X_embedded[203:,0], 
                    y=X_embedded[203:,1], 
                    color="red", 
                    label="modified"
                    )
    
    
    neighbours = 30
    ## extract good features
    fitter = umap.UMAP(n_neighbors=neighbours,
                       set_op_mix_ratio=0.1,
                       n_epochs = 1000,
                       learning_rate=0.5,
                       min_dist=0.5,
                       random_state=42,
                       n_components=2,
                       metric = 'chebyshev').fit(x.reshape((len(x)), 50)
                                                 )
    '''
    #fitter = umap.UMAP().fit(x.reshape((len(x)), 50),)
    
    '''
    sns.scatterplot(x=fitter.embedding_[:500,0], 
                    y=fitter.embedding_[:500,1], 
                    color="blue", 
                    label="No modified"
                    )
    
    sns.scatterplot(x=fitter.embedding_[500:,0], 
                    y=fitter.embedding_[500:,1], 
                    color="red", 
                    label="modified"
                    )
    
    test_data = fitter.transform(np.concatenate((no_mod[500:600], mod[500:600])))
    
    sns.scatterplot(x=test_data[:100,0], 
                    y=test_data[:100,1], 
                    color="blue", 
                    label="No modified"
                    )
    
    sns.scatterplot(x=test_data[100:,0], 
                    y=test_data[100:,1], 
                    color="red", 
                    label="modified"
                    )
    '''
    
    model_LocalOutlierFactor = LocalOutlierFactor(
                n_neighbors=20, contamination=0.2, novelty=True, leaf_size=10)
    
    # define model
    #model_svm = svm.OneClassSVM(nu=0.2, gamma='scale')
    
    model_isolation =  IsolationForest(contamination=0.2,
                                       random_state=42,
                                       behaviour='new',
                                       n_estimators=100,
                                       )
    
    model_EllipticEnvelope = EllipticEnvelope(contamination=0.2,
                                              support_fraction=1)
    '''
    f.write('LOF'+'\n')
    calculate_accuracies(model_LocalOutlierFactor, test_data, fitter, number, 100, f, LOF)
    
    f.write('Isolation forest'+'\n')
    calculate_accuracies(model_isolation, test_data, fitter, number, 100, f, iso)
    
    f.write('Envelop'+'\n')
    calculate_accuracies(model_EllipticEnvelope, test_data, fitter, number, 100, f, envelop)
    
    '''
    ### raw data 
    test_data = np.concatenate((no_mod[500:600], mod[500:600])))
    
    model_LocalOutlierFactor = LocalOutlierFactor(
                n_neighbors=20, contamination=0.2, novelty=True, leaf_size=10)
    
    # define model
    
    model_isolation =  IsolationForest(contamination=0.2,   
                                       random_state=42,
                                       behaviour='new',
                                       )
    
    model_EllipticEnvelope = EllipticEnvelope(contamination=0.2,
                                              support_fraction=0.8)
    
    f.write('LOF'+'\n')
    calculate_accuracies_raw(model_LocalOutlierFactor, test_data, x, number, 100, f)
    f.write('Isolation forest'+'\n')
    calculate_accuracies_raw(model_isolation, test_data, x, number, 100, f)
    f.write('Envelop'+'\n')
    calculate_accuracies_raw(model_EllipticEnvelope, test_data, x, number, 100, f)
    '''
f.close()
