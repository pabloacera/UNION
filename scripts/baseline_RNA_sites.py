#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:44:21 2020

@author: labuser
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:33:08 2020

@author: labuser

This sript will test the baseline of one-call methods using pUC19 modified and no modified reads
with RNA
"""

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import umap
import os

motif_WT_path = ['/media/labuser/Data/nanopore/S_cerevisiae_comb/WT/chr14_572034_np.npy',
                 '/media/labuser/Data/nanopore/S_cerevisiae_comb/WT/chr2_528304_np.npy',
                 '/media/labuser/Data/nanopore/S_cerevisiae_comb/WT/chr3_275229_np.npy']

motif_KO_path = ['/media/labuser/Data/nanopore/S_cerevisiae_comb/KO/chr14_572034_np.npy',
                '/media/labuser/Data/nanopore/S_cerevisiae_comb/KO/chr2_528304_np.npy',
                '/media/labuser/Data/nanopore/S_cerevisiae_comb/KO/chr3_275229_np.npy'
                ]


file_out = '/media/labuser/Data/nanopore/UNION/results/baseline_RNA_sites'
f = open(file_out, "w")


for i in range(len(motif_WT_path)):
    
    f.write(os.path.split(motif_WT_path[i])[1]+'\n')
    
    
    motif_KO = np.load(motif_KO_path[i])
    motif_WT = np.load(motif_WT_path[i])
    
    len_KO = len(motif_KO)
    len_WT = len(motif_WT)
    
    x = np.concatenate((motif_KO, motif_WT))
    
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
    
    model_LocalOutlierFactor = LocalOutlierFactor(
                    n_neighbors=20, contamination=0.2, novelty=True, leaf_size=10)
    
    model_isolation =  IsolationForest(contamination=0.2,
                                       random_state=42,
                                       behaviour='new',
                                       n_estimators=100,
                                       )
    
    model_EllipticEnvelope = EllipticEnvelope(contamination=0.2,
                                              support_fraction=1)
    
    f.write('Envelop'+'\n')
    
    # Training  the model
    model_EllipticEnvelope.fit(fitter.embedding_[:len_KO])
    
    # Measure the false positives
    acc = []
    for i in range(len_KO):
        total_training = fitter.embedding_[:len_KO]
        training = np.delete(total_training, i, axis=0)
        model_EllipticEnvelope.fit(training)
        acc += model_EllipticEnvelope.predict(total_training[i].reshape((1,4))).tolist()
    
    errors = acc.count(-1)
    
    print('Accuracy no mod '+str((len_KO-errors)/len_KO)+'\n')
    f.write('Accuracy no mod leave one out '+str((len_KO-errors)/len_KO)+'\n')
    
    # From the WT measure the stoichiometry
    prediction_WT = model_EllipticEnvelope.predict(fitter.embedding_[len_KO:])
    predicted_mod = len(prediction_WT[prediction_WT == -1])
    print('stoichiometry : '+str(predicted_mod/len_WT))
    f.write('stoichiometry : '+str(predicted_mod/len_WT)+'\n')
    
    #=============
    
    
    f.write('ISO'+'\n')
    
    # Training  the model
    model_isolation.fit(fitter.embedding_[:len_KO])
    
    # Measure the false positives
    acc = []
    for i in range(len_KO):
        total_training = fitter.embedding_[:len_KO]
        training = np.delete(total_training, i, axis=0)
        model_isolation.fit(training)
        acc += model_isolation.predict(total_training[i].reshape((1,4))).tolist()
    
    errors = acc.count(-1)
    
    print('Accuracy no mod '+str((len_KO-errors)/len_KO)+'\n')
    f.write('Accuracy no mod leave one out '+str((len_KO-errors)/len_KO)+'\n')
    
    # From the WT measure the stoichiometry
    prediction_WT = model_isolation.predict(fitter.embedding_[len_KO:])
    predicted_mod = len(prediction_WT[prediction_WT == -1])
    print('stoichiometry : '+str(predicted_mod/len_WT))
    f.write('stoichiometry : '+str(predicted_mod/len_WT)+'\n')
    
    
    #=============
    
    
    f.write('LOF'+'\n')
    
    # Training  the model
    model_LocalOutlierFactor.fit(fitter.embedding_[:len_KO])
    
    # Measure the false positives
    acc = []
    for i in range(len_KO):
        total_training = fitter.embedding_[:len_KO]
        training = np.delete(total_training, i, axis=0)
        model_LocalOutlierFactor.fit(training)
        acc += model_LocalOutlierFactor.predict(total_training[i].reshape((1,4))).tolist()
    
    errors = acc.count(-1)
    
    print('Accuracy no mod '+str((len_KO-errors)/len_KO)+'\n')
    f.write('Accuracy no mod leave one out '+str((len_KO-errors)/len_KO)+'\n')
    
    # From the WT measure the stoichiometry
    prediction_WT = model_LocalOutlierFactor.predict(fitter.embedding_[len_KO:])
    predicted_mod = len(prediction_WT[prediction_WT == -1])
    print('stoichiometry : '+str(predicted_mod/len_WT))
    f.write('stoichiometry : '+str(predicted_mod/len_WT)+'\n')


f.close()



