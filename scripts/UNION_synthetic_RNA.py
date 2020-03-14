#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:02:54 2020

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

from sklearn import svm
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import umap


motif_mod = '/media/labuser/Data/nanopore/S_cerevisiae_comb/synthetic_mod_np.npy'
motif = '/media/labuser/Data/nanopore/S_cerevisiae_comb/synthetic_no_mod_np.npy'


motif_mod = '/media/labuser/Data/nanopore/Epinanot_IVT/mod/GGACC_35_np.npy'
motif = '/media/labuser/Data/nanopore/Epinanot_IVT/no_mod/GGACC_35_np.npy'


file_out = '/media/labuser/Data/nanopore/UNION/results/baseline_RNA_IVT'
f = open(file_out, "w")


f.write('IVT synthetic')
no_mod = np.load(motif)
mod = np.load(motif_mod)
                        
    
len_mod = len(mod)
len_no_mod = len(no_mod)
    
x = np.concatenate((no_mod,mod))

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

#fit the model
model_EllipticEnvelope.fit(fitter.embedding_[:len_no_mod])

f.write('envelop \n')

# predict and accuraacy for the training the training
prediction = model_EllipticEnvelope.predict(fitter.embedding_)
n_error_outliers = prediction[len_no_mod:][prediction[len_no_mod:] == 1].size
print('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')
f.write('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')

# maybe do leave one out with the no mod
acc = []
for i in range(len_mod):
    total_training = fitter.embedding_[:len_mod]
    training = np.delete(total_training, i, axis=0)
    model_EllipticEnvelope.fit(training)
    acc += model_EllipticEnvelope.predict(total_training[i].reshape((1,4))).tolist()

errors = acc.count(-1)

print('Ac training '+str((len_mod-errors)/len_mod)+'\n')
f.write('Ac training '+str((len_mod-errors)/len_mod)+'\n')


f.write('iso \n')
# fit the envelope
model_isolation.fit(fitter.embedding_[:len_no_mod])

# predict and accuraacy for the training the training
prediction = model_isolation.predict(fitter.embedding_)
n_error_outliers = prediction[len_no_mod:][prediction[len_no_mod:] == 1].size
print('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')
f.write('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')

# maybe do leave one out with the no mod
acc = []
for i in range(len_mod):
    total_training = fitter.embedding_[:len_mod]
    training = np.delete(total_training, i, axis=0)
    model_isolation.fit(training)
    acc += model_isolation.predict(total_training[i].reshape((1,4))).tolist()

errors = acc.count(-1)

print('Ac training '+str((len_mod-errors)/len_mod)+'\n')
f.write('Ac training '+str((len_mod-errors)/len_mod)+'\n')


f.write('LOF \n')
model_LocalOutlierFactor.fit(fitter.embedding_[:len_no_mod])

# predict and accuraacy for the training the training
prediction = model_LocalOutlierFactor.predict(fitter.embedding_)
n_error_outliers = prediction[len_no_mod:][prediction[len_no_mod:] == 1].size
print('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')
f.write('Ac mod testing '+str((len(prediction[len_no_mod:])-n_error_outliers)/len(prediction[len_no_mod:]))+'\n')

# maybe do leave one out with the no mod
acc = []
for i in range(len_mod):
    total_training = fitter.embedding_[:len_mod]
    training = np.delete(total_training, i, axis=0)
    model_LocalOutlierFactor.fit(training)
    acc += model_LocalOutlierFactor.predict(total_training[i].reshape((1,4))).tolist()

errors = acc.count(-1)

print('Ac training '+str((len_mod-errors)/len_mod)+'\n')
f.write('Ac training '+str((len_mod-errors)/len_mod)+'\n')
    
    
f.close()