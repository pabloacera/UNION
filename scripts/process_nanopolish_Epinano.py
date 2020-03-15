#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:20:34 2020

@author: labuser
take a file with the selected reads from nanopolish and process them
the negative sites the roght kmer is the one on the model_kmer not the reference

This script is not going to scale the samples to anything, as they are already scaled
by nanopolish. 
Will select the lines that have position that I want
"""

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from math import floor
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import seed
from tensorflow import set_random_seed    
import seaborn as sns
from scipy import stats
import re

def min_max_scale(signal):
    '''
    Function to scale the data
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    signal = signal.reshape((len(signal),1))
    signal = min_max_scaler.fit_transform(signal)
    return signal


def top_median(array, lenght):
    '''
    This function top an array until some specific lenght
    '''
    extra_measure = [np.median(array)]*(lenght-len(array))
    array += extra_measure
    return array


def smooth_event(raw_signal, lenght_events):
    '''
    smmoth the signal 
    '''
    raw_signal_events = []
    if len(raw_signal) < lenght_events:
        event = top_median(raw_signal, lenght_events)
        raw_signal_events = raw_signal_events + [event]
    else:
        division = floor(len(raw_signal)/lenght_events)
        new_event = []
        for i in range(0, len(raw_signal), division):
            new_event.append(np.median(raw_signal[i:i+1]))
            if len(new_event) == 10:
                break
            
        if len(new_event) < lenght_events:
            new_event = top_median(new_event, lenght_events)
        raw_signal_events = raw_signal_events + [new_event]

    return raw_signal_events

# Now I have the signal information from the site KO and WT
def MAD(raw_signal):
    return stats.median_absolute_deviation(raw_signal)


def plot_signals(KO, WT):
    '''
    function to plot a bucnh of KO and WT signals overlapping
    '''
    plt.figure(figsize=(20, 15))
    
    custom_lines = [Line2D([0], [0], color='black', lw=4, markersize=40),
                    Line2D([0], [0], color='red', lw=4, markersize=40)]
    
    for i in range(len(KO)):
        signal_KO = [item for sublist in KO[i] for item in sublist]
        try:
            signal_WT = [item for sublist in WT[i] for item in sublist]
        except:
            continue            
        plt.plot(signal_KO, color='black', label='KO')
        plt.plot(signal_WT, color='red', label='WT')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('Signal intensity', fontsize=30)
    
    X = [5, 15, 25, 35, 45]

    labels = ['G', 'G', 'A', 'C', 'A']
    
    # You can specify a rotation for the tick labels in degrees or with keywords.
    plt.xticks(X, labels, fontsize=30)
    
    plt.legend(custom_lines,
               ['KO','WT'],
               fancybox=True,
               framealpha=1, 
               shadow=True, 
               borderpad=1,
               fontsize=20
               )
    
    plt.show()
    
    return True

    
def plot_signals_sd(KO_signal_filtered, WT_signal_filtered, save=None):
    '''
    This function wil plot the median and sd of a bunch of signals
    '''
    # first build the whole signals with medians
    sns.set_style('darkgrid')
    
    KO_signals_medians = []
    WT_signals_medians = []
    for i in range(len(KO_signal_filtered)):
        KO_signals_medians.append([np.median(j) for j in KO_signal_filtered[i]])
    
    for i in range(len(WT_signal_filtered)):
        WT_signals_medians.append([np.median(j) for j in WT_signal_filtered[i]])
    
    # I think seaborn needs just the numbers and it will calculate the median itself
    event1_KO = [i[0] for i in KO_signals_medians]  
    event2_KO = [i[1] for i in KO_signals_medians] 
    event3_KO = [i[2] for i in KO_signals_medians] 
    event4_KO = [i[3] for i in KO_signals_medians] 
    event5_KO = [i[4] for i in KO_signals_medians] 
    
    events_KO = event1_KO + event2_KO + event3_KO + event4_KO + event5_KO            
    nucleotides_KO = len(event1_KO)*[1]+len(event2_KO)*[2]+len(event3_KO)*[3]+len(event4_KO)*[4]+ len(event5_KO)*[5]
    KO_df = pd.DataFrame({'events' : nucleotides_KO, 
                          'signal' : events_KO})
    
    event1_WT = [i[0] for i in WT_signals_medians]  
    event2_WT = [i[1] for i in WT_signals_medians] 
    event3_WT = [i[2] for i in WT_signals_medians] 
    event4_WT = [i[3] for i in WT_signals_medians] 
    event5_WT = [i[4] for i in WT_signals_medians]
     
    events_WT = event1_WT + event2_WT + event3_WT + event4_WT + event5_WT            
    nucleotides_WT = len(event1_WT)*[1]+len(event2_WT)*[2]+len(event3_WT)*[3]+len(event4_WT)*[4]+ len(event5_WT)*[5]
    WT_df = pd.DataFrame({'events' : nucleotides_WT, 
                          'signal' : events_WT})
    
    plt.figure(figsize=(20, 11))
    ax = sns.lineplot(x="events", y="signal", ci='sd', estimator=np.median, data=WT_df, color='red')
    ax2 = sns.lineplot(x="events", y="signal", ci='sd', estimator=np.median, data=KO_df, color='blue')
    ax2.lines[0].set_linestyle("--")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Signal intensity', fontsize=30)
    X = [1, 2, 3, 4, 5]
    labels = ['T','C','C','A','G']
    # You can specify a rotation for the tick labels in degrees or with keywords.
    plt.xticks(X, labels, fontsize=30)
    
    custom_lines = [Line2D([0], [0], color='blue', lw=4, markersize=40),
                    Line2D([0], [0], color='red', lw=4, markersize=40, linestyle='--')]
    
    plt.legend(custom_lines,
               ['KO','WT'],
               fancybox=True,
               framealpha=1, 
               shadow=True, 
               borderpad=1,
               fontsize=20
               )
    if save:
        plt.savefig(save+'.pdf',
                    format='pdf',
                    dpi=1200)
        
    return True

    
def plot_signals_10(KO_signal_filtered_np, WT_signal_filtered_np, save=None):
    '''
    This function wil plot the median and sd of a bunch of signals
    '''
    # first build the whole signals with medians
    sns.set_style('darkgrid')

    KO_signal = KO_signal_filtered_np.transpose().ravel()
    KO_signal_index = np.arange(1, 51)
    KO_signal_index = np.repeat(KO_signal_index, KO_signal_filtered_np.shape[0])
    
    KO_df = pd.DataFrame({'events' : KO_signal_index, 
                          'signal' : KO_signal})
    
    
    WT_signal = WT_signal_filtered_np.transpose().ravel()
    WT_signal_index = np.arange(1, 51)
    WT_signal_index = np.repeat(WT_signal_index, WT_signal_filtered_np.shape[0])
    
    WT_df = pd.DataFrame({'events' : WT_signal_index, 
                          'signal' : WT_signal})
    plt.figure(figsize=(20, 11))
    ax = sns.lineplot(x="events", y="signal", ci='sd', estimator=np.median, data=WT_df, color='red')
    ax2 = sns.lineplot(x="events", y="signal", ci='sd', estimator=np.median, data=KO_df, color='blue')
    ax2.lines[0].set_linestyle("--")
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('Signal intensity', fontsize=30)
    X = [5, 15, 25, 35, 45]
    labels = ['G','G','A','C','C']
    # You can specify a rotation for the tick labels in degrees or with keywords.
    plt.xticks(X, labels, fontsize=30)
    
    custom_lines = [Line2D([0], [0], color='blue', lw=4, markersize=40),
                    Line2D([0], [0], color='red', lw=4, markersize=40, linestyle='--')]
    
    plt.legend(custom_lines,
               ['NO MOD','MOD'],
               fancybox=True,
               framealpha=1, 
               shadow=True, 
               borderpad=1,
               fontsize=20
               )
    if save:
        plt.savefig(save+'.pdf',
                    format='pdf',
                    dpi=1200)
    return True



def extract_signal_forward(positions, file_complete, out):
    '''
    function to extract the right signals from the file
    '''
    
    counter = 0
    for file in pd.read_csv(file_complete, sep='\t', 
                             chunksize=1000000):
        counter +=1
        contig_file = file[file['contig'] == 'cc6m_2244_t7_ecorv']
        position_file = contig_file[contig_file['position'].isin(positions['cc6m_2244_t7_ecorv'])]
        if counter == 1:
            position_file.to_csv(out, sep='\t')
        else:
            position_file.to_csv(out, sep='\t', mode='a', header=False)
        
        contig_file = file[file['contig'] == 'cc6m_2459_t7_ecorv']
        position_file = contig_file[contig_file['position'].isin(positions['cc6m_2459_t7_ecorv'])]
        position_file.to_csv(out, sep='\t', mode='a', header=False)
        
        contig_file = file[file['contig'] == 'cc6m_2595_t7_ecorv']
        position_file = contig_file[contig_file['position'].isin(positions['cc6m_2595_t7_ecorv'])]
        position_file.to_csv(out, sep='\t', mode='a', header=False)


        contig_file = file[file['contig'] == 'cc6m_2709_t7_ecorv']
        position_file = contig_file[contig_file['position'].isin(positions['cc6m_2709_t7_ecorv'])]
        position_file.to_csv(out, sep='\t', mode='a', header=False)
    return True



if __name__ == '__main__':
    
    KO_path = '/media/labuser/CIMA_maite/Epinano_IVT/non_mod/Non_mod_rep1_eventalign.txt'
    WT_path = '/media/labuser/CIMA_maite/Epinano_IVT/mod_rep2/nanopolish/mod_rep2_eventalign.txt'
    
    
    Epinano_ref = '/media/labuser/CIMA_maite/Epinano_IVT/GSE124309_FASTA_sequences_of_Curlcakes.fasta'
    reference = {}
    sequence= ''
    with open(Epinano_ref, 'r') as file_ref:
        for line in file_ref:
            line = line.rstrip()
            if line[0] == '>':
                if not sequence:
                    contig = line[1:]
                    continue
                else:
                    reference[contig] = sequence
                    contig = line[1:]
                    sequence = ''
                    continue
            line = line.rstrip()
            sequence +=line
        reference[contig] = sequence
    
    positions = {}
    for contig in reference:
        index = [m.start() for m in re.finditer('GGACC', reference[contig])]
        final_index = []
        
        # Get rid of positions where there is another A in the neighbouring places
        for i in range(0,len(index)):
            if reference[contig][index[i]-1] == 'A' or reference[contig][index[i]+5] == 'A':
                continue
            else:
                final_index.append(index[i])
        positions[contig] = final_index
        
        
    for contig in reference:
        index = [m.start() for m in re.finditer('GGACT', reference[contig])]
        final_index = []
        # Get rid of positions where there is another A in the neighbouring places
        for i in range(0,len(index)):
            if reference[contig][index[i]-1] == 'A' or reference[contig][index[i]+5] == 'A':
                continue
            else:
                final_index.append(index[i])
        positions[contig] =  positions[contig] + final_index
    
    #complete with all the positions from the 5-mers and also clean the name of the k-mers
    all_positions = {}
    for contig in positions:
        list_f = []
        for i in positions[contig]:
            list_f += list(range(i-3,i+3))
        all_positions[contig.split(' ')[0]] = list_f
  
    # We are modeling kmers so I am taking all kmers that contain the modified A
    KO_signal = extract_signal_forward(all_positions, KO_path, '/media/labuser/Data/nanopore/Epinanot_IVT/no_mod/eventalign_no_mod_2.txt')
    
    '''
    # it is a bummer that some of the eventalign dont span through all the reference k-mer
    # so I have to filter them out
    KO_signal_filtered = []
    for i in KO_signal:
        if len(i) == 5:
            KO_signal_filtered.append(i)
    
    WT_signal_filtered = []
    for i in WT_signal:
        if len(i) == 5:
            WT_signal_filtered.append(i)

    
    #plot_signals(KO_signal_filtered, WT_signal_filtered)
    
    ### plot signals with deviation from median
    # Now every 10 measurements event is converted into 1 median
    # and the different from other medians is the one represented in the plot
      
    plot_signals_sd(KO_signal_filtered, 
                    WT_signal_filtered)
                    '/media/labuser/Seagate Expansion Drive/pablo_hec293_IVT/ENST00000301821.10/plots/ENST00000301821_343_sd')
    
    
    #####
    # Save the signals in numpy objects
    #####
    
    KO_signal_filtered_np = []
    WT_signal_filtered_np = []
    
    #Convert to numpy
    for i in range(len(KO_signal_filtered)):
        KO_signal_filtered_np.append([item for sublist in KO_signal_filtered[i] for item in sublist])
    
    for i in range(len(WT_signal_filtered)):
        WT_signal_filtered_np.append([item for sublist in WT_signal_filtered[i] for item in sublist])
    
    KO_signal_filtered_np = np.array(KO_signal_filtered_np)
    WT_signal_filtered_np = np.array(WT_signal_filtered_np)
    
    #### Plot the signals with 10 values each event
    plot_signals_10(KO_signal_filtered_np, 
                    WT_signal_filtered_np
                    '/media/labuser/CIMA_maite/Epinano_IVT/plots/GGACC_35'
                    )
    # saving the numpy signals 
    save_np = '/media/labuser/Data/nanopore/Epinanot_IVT'
    np.save(save_np+'/no_mod/GGACC_35_np', KO_signal_filtered_np)
    np.save(save_np+'/mod/GGACC_35_np', WT_signal_filtered_np)
    
    '''


