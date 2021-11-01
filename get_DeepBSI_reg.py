#!/usr/bin/env python

"""
Usage:
nohup python get_deepBSI_reg.py  >> log_deepBSI_reg.txt 2>&1 &
"""

import numpy as np
import pandas as pd
import sys
import errno
import argparse
import pickle

from pybedtools import BedTool, Interval
import pyfasta
import parmap
import itertools
import pyBigWig
import random
import os
import scipy
import scipy.stats
import time
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import model_from_json
import tensorflow as tf

#keras
from keras import layers
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input,BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed

##define the CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

#set the gpu fraction to 0.2
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# sess = tf.compat.v1.Session(config=config)

start_time_script = time.time()
print('beginning time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

##basic functions
def check_file_ifexists(file_name):
    if os.path.exists(file_name):
        pass
    else:
        print('\nWarning!!! file {0} does not exists!\n'.format(file_name))

def get_running_times(start_time, stop_time):
    times_seconds = stop_time - start_time
    times_minutes = int(times_seconds / 60)
    times_hours = round(times_minutes / 60, 1)
    return int(times_seconds), times_minutes, times_hours

output_dir = 'output_deepBSI_reg/'

if os.path.exists(output_dir):
    print('outputdir exists!')
else:
    os.makedirs(output_dir)

chromslists_all = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']
# chromslists_all = [ x for x in chromslists_all if x not in ['chr1','chr8', 'chr21'] ]
chromslists_test = ['chr1','chr8', 'chr21']
chromslists_train_valid = [ x for x in chromslists_all if x not in  chromslists_test ]
# random.seed(0)
# random.shuffle(chromslists_train_valid)
chromslists_valid = chromslists_train_valid[-2:]
chromslists_train = [ x for x in chromslists_train_valid if x not in chromslists_valid ]

# chromslists_train = ['chr19']
# chromslists_valid = ['chr22']
# chromslists_test = ['chr21']
# chromslists_train_valid = chromslists_train + chromslists_valid
# chromslists_all = chromslists_train + chromslists_valid + chromslists_test

genome_fasta_file = '/cmachdata1/zhangpeng/genomes/hg19.fa'
genome_sizes_file = '/cmachdata1/zhangpeng/genomes/hg19.autoX.chrom.sizes'
genome_sizes = pd.read_csv(genome_sizes_file, sep='\t', header=None)
genome_sizes = genome_sizes.values.tolist()
dict_genome_sizes = dict(genome_sizes)
# genome_sizes = BedTool([ [x[0], 0, int(x[1]) ] for x in genome_sizes])


##features 
##features 
##features 
print('preprocessing the basic features...')
#1. feature seq onehot 
print('features seq onehot...')
def get_onehot_chrom(chrom): 
    fasta = pyfasta.Fasta(genome_fasta_file)
    seq_chrom = str(fasta[chrom]).upper()
    seq_array = np.array(list(seq_chrom)).reshape(-1,1)
    acgt = np.array(['A','C','G','T'])
    seq_onehot = seq_array == acgt
    return seq_onehot

dict_seqonehot = {}
for each in chromslists_all:
    print('preprocessing ' + each + ' ...')
    onehot_chrom = get_onehot_chrom(each)
    dict_seqonehot[each] = onehot_chrom


tfs_all = [ 'CTCF', 'EP300', 'JUN', 'MYC', 'POLR2A', 'RAD21', 'REST', 'TAF1', 'GABPA', 'SIN3A']
# 'CEBPB', 'JUND'
cell_all = ['HepG2','K562','A549','GM12878','MCF-7' ,'HeLa-S3' ,'H1' ]

results = []
for tf_name in tfs_all:
    # index_tf = 0
    # tf_name = tfs_all[index_tf]
    print('processing... ' + tf_name)
    data_tfs_path = '/cmachdata1/zhangpeng/encoderawdata/tfs2/'
    data_files = os.listdir(data_tfs_path)
    data_files = [ x for x in data_files if (x.split('_')[0] == tf_name ) ]
    cell_tfs = list(set([ x.split('_')[1] for x in data_files ]))
    data_narrow_tfs = [ x for x in data_files if x.split('.')[0].endswith('narrowpeak') ]
    data_broad_tfs = [ x for x in data_files if x.split('.')[0].endswith('broadpeak') ]
    data_bw_tfs = [x for x in data_files if x.endswith('bigwig')]

    for cell_name in cell_tfs:
        # cell_name = cell_tfs[0]
        cell_name_rest = [ x for x in cell_tfs if x!=cell_name ]
        data_pos = [ x for x in data_narrow_tfs if x.split('_')[1] == cell_name ]
        data_pos = BedTool(data_tfs_path + data_pos[0])
        num_peak = len(data_pos)
        if str(data_pos[0][8]) == '-1':
            data_pos = [ [x[0],int(x[1]), int(x[2]), float(x[6]), int(x[9])] for x in data_pos ]
        else:
            data_pos = [ [x[0],int(x[1]), int(x[2]), float(x[6]), int(x[9])] for x in data_pos if float(x[8]) >= 2 ]

        if len(data_pos) <= 1000:
            continue
        ##plot the peak len distribution
        data_pos_peak_len = [ int(x[2]-x[1]) for x in data_pos ]
        peak_most_common = Counter(data_pos_peak_len).most_common(1)[0][0]
        L = int(np.ceil(peak_most_common/100)*100)
        flankinglen = int(L / 2)

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(7,6))
        # plt.hist(data_pos_peak_len, 30)
        # plt.xlabel('The length of the peak')
        # plt.ylabel('The count')
        # plt.title( 'The distribution of ChIPseq peak length of ' + tf_name +' in ' +cell_name)
        # plt.savefig(output_dir + 'plot.narrowpeak.length.distribution.' + tf_name + '.' + cell_name +'.pdf')

        ##neg 1 tfbs in other cells but not in target cell
        #neg 2 no tfbs
        data_broad_files = [ x for x in data_broad_tfs if x.split('_')[1] == cell_name ]
        data_bw_files = [ x for x in data_bw_tfs if x.split('_')[1] == cell_name ]
        if len(data_broad_files) >0:
            assert len(data_broad_files) == 2
            data1 = BedTool( data_tfs_path + data_broad_files[0])
            data2 = BedTool( data_tfs_path + data_broad_files[1])
            data1_only = data1.intersect(data2, v = True)
            data2_only = data2.intersect(data1, v = True)
            data_overlap = data1.intersect(data2, wa=True, wb=True)
            data_overlap = [ [ x[0], min(int(x[1]), int(x[10])), max(int(x[2]), int(x[11])) ] for x in data_overlap ]
            data1_only = [ [x[0], int(x[1]), int(x[2]) ] for x in data1_only ]
            data2_only = [ [x[0], int(x[1]), int(x[2]) ] for x in data2_only ]
            data_broad = data1_only + data2_only + data_overlap
        else:
            data_broad = []

        def get_posneg_data( data_pos_input, data_broad_input, chromslists_fun):
            # data_pos_input = data_pos
            # data_broad_input = data_broad
            # chromslists_fun = chromslists_train
            data_output = []
            for each_chrom in chromslists_fun:
                # each_chrom = chromslists_fun[0]
                data_pos_chrom = [ x for x in data_pos_input if x[0] == each_chrom ]
                data_pos_chrom_psudo = [ [x[0], x[1]+x[4]-flankinglen, x[1]+x[4]+flankinglen, 0, flankinglen ] for x in data_pos_input if x[0] == each_chrom ]
                genome_sizes_chrom = BedTool([ [x[0],0, int(x[1])] for x in genome_sizes if x[0] == each_chrom ])
                data_broad_chrom = BedTool([ x for x in data_broad_input if x[0]==each_chrom ])
                data_neg_chrom = BedTool(data_pos_chrom_psudo).shuffle( g=genome_sizes_file, incl= genome_sizes_chrom.fn, excl=data_broad_chrom.fn, chromFirst=True, chrom=True, noOverlapping=True, seed = 1 )
                data_neg_chrom = [ list(x) for x in data_neg_chrom ]
                data_chrom = data_pos_chrom + data_neg_chrom
                data_output.extend( data_chrom )
            return data_output

        data_train = get_posneg_data(data_pos, data_broad, chromslists_train)
        data_valid = get_posneg_data(data_pos, data_broad, chromslists_valid)
        data_test = get_posneg_data(data_pos, data_broad, chromslists_test)

        random.shuffle(data_train)
        random.shuffle(data_valid)
        random.shuffle(data_test)

        ##seq model
        def get_seq_features_and_label(data_input):
            x_seq_fwd = np.zeros((len(data_input), L, 4), dtype=np.float32)
            y = np.zeros((len(data_input), 1), dtype=np.float32)
            for index_data, each_peak in enumerate( data_input ):
                chrom, start, stop, signal_value, summit_point = each_peak
                start = int(start)
                stop = int(stop)
                summit_point = int(summit_point)
                seqlen = int(stop) - int(start) 
                signal_value = float(signal_value)
                y[index_data] = signal_value
                if seqlen == L:
                    startfinal = start
                    stopfinal = stop
                    seq_onehot = dict_seqonehot[chrom][startfinal:stopfinal]
                elif seqlen < L:
                    left = int( (L - seqlen) / 2) 
                    right = L - left - seqlen 
                    seq_onehot= dict_seqonehot[chrom][start:stop]
                    seq_onehot = np.concatenate([ np.zeros((left, 4)),seq_onehot, np.zeros((right, 4)) ],axis=0)
                elif seqlen > L:
                    if ( summit_point + L / 2) >= seqlen:
                        stopfinal = seqlen
                        startfinal = seqlen - L 
                    else:
                        if (summit_point - L / 2) < 1:
                            startfinal = 1
                        else:
                            startfinal = summit_point - L / 2
                    startfinal = int(start + startfinal)
                    stopfinal = int(startfinal + L - 1)
                    seq_onehot = dict_seqonehot[chrom][ startfinal-1 : stopfinal]
                x_seq_fwd[index_data] = seq_onehot
            x_seq_rev = x_seq_fwd[:,::-1,::-1]
            y = y
            return x_seq_fwd, x_seq_rev, y

        x_seq_fwd_train, x_seq_rev_train, y_train = get_seq_features_and_label(data_train)
        x_seq_fwd_valid, x_seq_rev_valid, y_valid = get_seq_features_and_label(data_valid)
        x_seq_fwd_test, x_seq_rev_test, y_test = get_seq_features_and_label(data_test)

        ##annotation model
        #  tf chipseq
        chipseq_files = [ data_tfs_path + x for x in data_bw_tfs if x.split('_')[1] != cell_name ]
        num_chipseq = len(chipseq_files)
        print(num_chipseq)

        annotation_files = chipseq_files
        num_annotation = len(annotation_files)
        print(num_annotation)

        def get_annotation_features(data_input, annotation_files_input):
            x_seq_fwd = np.zeros((len(data_input), L, 4), dtype=np.float32)
            x_annotation_fwd = np.zeros((len(data_input), L,  num_annotation), dtype=np.float32)
            # y = np.zeros((len(data_input), 1), dtype=np.float32)
            for index_bw, each_bw in enumerate(annotation_files_input):
                bigwig = pyBigWig.open( each_bw )
                for index_data, each_peak in enumerate( data_input ):
                    chrom, start, stop, signal_value, summit_point = each_peak
                    start = int(start)
                    stop = int(stop)
                    summit_point = int(summit_point)
                    seqlen = int(stop) - int(start) 
                    # y[index_data] = signal_value
                    if seqlen == L:
                        startfinal = start
                        stopfinal = stop
                        # seq_onehot = dict_seqonehot[chrom][startfinal:stopfinal]
                        #epi
                        sample_bigwig = np.array(bigwig.values(chrom, startfinal, stopfinal))        
                        sample_bigwig[np.isnan(sample_bigwig)] = 0
                        x_annotation_fwd[index_data, :, index_bw] = sample_bigwig
                    elif seqlen < L:
                        left = int( (L - seqlen) / 2) 
                        right = L - left - seqlen 
                        # seq_onehot= dict_seqonehot[chrom][start:stop]
                        # seq_onehot = np.concatenate([ np.zeros((left, 4)),seq_onehot, np.zeros((right, 4)) ],axis=0)
                        #epi
                        sample_bigwig = np.array(bigwig.values(chrom, start, stop))
                        sample_bigwig[np.isnan(sample_bigwig)] = 0
                        sample_bigwig = [0] * left + list(sample_bigwig) + [0] * right
                        # print(sample_bigwig, len(sample_bigwig), left, right)
                        x_annotation_fwd[index_data, :, index_bw] = sample_bigwig
                    elif seqlen > L:
                        if ( summit_point + L / 2) >= seqlen:
                            stopfinal = seqlen
                            startfinal = seqlen - L 
                        else:
                            if (summit_point - L / 2) < 1:
                                startfinal = 1
                            else:
                                startfinal = summit_point - L / 2
                        startfinal = int(start + startfinal)
                        stopfinal = int(startfinal + L - 1)
                        # seq_onehot = dict_seqonehot[chrom][ startfinal-1 : stopfinal]
                        #epi
                        sample_bigwig = np.array(bigwig.values(chrom, startfinal-1, stopfinal))
                        sample_bigwig[np.isnan(sample_bigwig)] = 0
                        x_annotation_fwd[index_data, :, index_bw] = sample_bigwig
                bigwig.close()
                # x_seq_fwd[index_data] = seq_onehot
            # x_seq_rev = x_seq_fwd[:,::-1,::-1]
            x_annotation_rev = x_annotation_fwd[:,::-1,:]
            # x_fwd = np.concatenate([x_seq_fwd, x_annotation_fwd], axis=-1)
            # x_rev = np.concatenate([x_seq_rev, x_annotation_rev], axis=-1)
            # y = y
            return x_annotation_fwd, x_annotation_rev

        x_annotation_fwd_train, x_annotation_rev_train  = get_annotation_features(data_train, annotation_files)
        x_annotation_fwd_valid, x_annotation_rev_valid  = get_annotation_features(data_valid, annotation_files)
        x_annotation_fwd_test, x_annotation_rev_test  = get_annotation_features(data_test, annotation_files)


        print('\n\n\nbuilding the model and train')
        print('begin time:', time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        learning_rate = 0.001
        patience = 5
        epochs = 50
        num_kernels = 64
        w = 26
        w2 = int(w / 2)
        dropout_rate = 0.5
        num_recurrent = 32
        num_dense1 = 256
        num_dense2 = 256
        batch_size = 256

        def get_output(input_layer, shared_layers):
            output = input_layer
            for layer_each in shared_layers:
                output = layer_each(output)
            return output

        ###seq model
        shared_layers_seq = [  Convolution1D(filters=num_kernels, kernel_size=w, activation='relu'),  
        MaxPooling1D(pool_size=int(w2), strides=int(w2), padding='valid'), 
        Bidirectional(LSTM(num_recurrent, recurrent_dropout = 0.2, dropout = 0.2, return_sequences=True)),  
        BatchNormalization(),
        Flatten(), 
        Dense(num_dense1, activation='relu') ]

        forward_input_seq = Input(shape=(L, 4), dtype='float32')
        reverse_input_seq = Input(shape=(L, 4), dtype='float32')
        forward_output_seq = get_output(forward_input_seq, shared_layers_seq)
        reverse_output_seq = get_output(reverse_input_seq, shared_layers_seq)
        output_seq = layers.concatenate([forward_output_seq, reverse_output_seq], axis=-1)
        output_seq = Dense(num_dense2)(output_seq)

        #ann module
        shared_layers_ann = [  Convolution1D(filters=num_kernels, kernel_size=w, activation='relu'),  
        MaxPooling1D(pool_size=int(w2), strides=int(w2), padding='valid'), 
        Bidirectional(LSTM(num_recurrent, recurrent_dropout = 0.2, dropout = 0.2, return_sequences=True)),  
        BatchNormalization(),
        Flatten(), 
        Dense(num_dense1, activation='relu') ]

        forward_input_ann = Input(shape=(L, num_annotation), dtype='float32')
        reverse_input_ann = Input(shape=(L, num_annotation), dtype='float32')
        forward_output_ann = get_output(forward_input_ann, shared_layers_ann)
        reverse_output_ann = get_output(reverse_input_ann, shared_layers_ann)
        output_ann = layers.concatenate([forward_output_ann, reverse_output_ann], axis=-1)
        output_ann = Dense(num_dense2)(output_ann)

        ##joint model
        merged_features = layers.concatenate([output_seq, output_ann], axis=-1)
        output = Dense(num_dense2, activation='relu')(merged_features)
        output = Dropout(dropout_rate)(output)
        output = Dense(1)(output)

        model = Model([forward_input_seq, reverse_input_seq, forward_input_ann, reverse_input_ann ], output)
        print(model.summary())

        #save model
        model_json = model.to_json()
        output_json_file = open(output_dir + '/model.reg.' + str(tf_name) + '.' + cell_name + '.json', 'w')
        output_json_file.write(model_json)
        output_json_file.close()

        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam

        print('Compiling model')
        model.compile('rmsprop', 'mse', metrics=['mae'])
        # model.compile(Adam(lr=learning_rate), 'binary_crossentropy', metrics=['acc'])

        checkpointer = ModelCheckpoint(filepath=output_dir + '/best.model.reg.' + str(tf_name) + '.' +cell_name +'.hdf5', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
        reducelronplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        history = model.fit([x_seq_fwd_train, x_seq_rev_train, x_annotation_fwd_train, x_annotation_rev_train], y_train , epochs=epochs, batch_size=batch_size, validation_data= ([x_seq_fwd_valid, x_seq_rev_valid, x_annotation_fwd_valid, x_annotation_rev_valid], y_valid), callbacks=[checkpointer, earlystopper, reducelronplateau])

        print('regression model stop time :', time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))

        ##predict
        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score
        import scipy
        from keras.models import model_from_json
        model_json_file = open(output_dir + '/model.reg.' + str(tf_name) +  '.' +cell_name +'.json', 'r')
        model_json = model_json_file.read()
        model = model_from_json(model_json)
        model.load_weights( output_dir + '/best.model.reg.' + str(tf_name) + '.' + cell_name +'.hdf5')

        # results = np.zeros( (len(y_test), 2), dtype='float32' )
        y_predict = model.predict([x_seq_fwd_test, x_seq_rev_test, x_annotation_fwd_test, x_annotation_rev_test])
        r2_score(y_test, y_predict)
        y_predict_list = y_predict.reshape(1,-1).tolist()[0]
        y_test_list = 1*y_test.reshape(1,-1).tolist()[0]

        ##write to txt
        results_test = pd.DataFrame(data_test)
        results_test.columns = ['chrom', 'start', 'stop', 'signal_value', 'summit_point']
        results_test['predicted'] = y_predict_list
        results_test.to_csv( output_dir + 'results_test_'+ tf_name + '.' +cell_name + '.txt', sep='\t', header=True, index=None)

        #scatter plot
        y_predict_max = max(y_predict_list)
        y_test_max = max(y_test_list)
        y_plot_max = max(y_predict_max, y_test_max)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,6))
        plt.scatter(y_test_list, y_predict_list, s=1)
        # plt.text(0.05 , 0.95, 'r2 score: ' + str(round(r2_score(1*y_test, y_predict),3 )) )
        plt.text(3, y_plot_max - 20,  'pearsonr: ' + str(round(scipy.stats.pearsonr( y_predict_list, y_test_list )[0],3) ),fontsize=15)
        plt.text(3,y_plot_max-40, 'spearmanr: ' + str(round(scipy.stats.spearmanr( y_predict_list, y_test_list )[0],3)),fontsize=15)
        plt.plot([0,y_plot_max],[0,y_plot_max], 'r--')
        plt.xlim(-5,y_plot_max)
        plt.ylim(-5,y_plot_max)
        plt.xlabel('Experimental intensity')
        plt.ylabel('Predicted tf intensity')
        plt.title( 'The correlation of ' +tf_name +' intensity in '+ cell_name)
        plt.savefig(output_dir + 'plot.scatter.' + tf_name + '.' +cell_name + '.pdf')

        #no labels
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,6))
        plt.scatter(y_test_list, y_predict_list, s=1)
        # plt.text(0.05 , 0.95, 'r2 score: ' + str(round(r2_score(1*y_test, y_predict),3 )) )
        # plt.text(3, y_plot_max - 20,  'pearsonr: ' + str(round(scipy.stats.pearsonr( y_predict_list, y_test_list )[0],3) ),fontsize=15)
        # plt.text(3,y_plot_max-40, 'spearmanr: ' + str(round(scipy.stats.spearmanr( y_predict_list, y_test_list )[0],3)),fontsize=15)
        plt.plot([0,y_plot_max],[0,y_plot_max], 'r--')
        plt.xlim(-5,y_plot_max)
        plt.ylim(-5,y_plot_max)
        plt.xticks([],[])
        plt.yticks([],[])
        # plt.xlabel('Experimental intensity')
        # plt.ylabel('Predicted tf intensity')
        # plt.title( 'The correlation of ' +tf_name +' intensity in '+ cell_name)
        plt.savefig(output_dir + 'nolabel.plot.scatter.' + tf_name + '.' +cell_name + '.pdf')

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test_list, y_predict_list)
        mse = round(mse,1)

        pearsonr = round(scipy.stats.pearsonr( y_predict_list, y_test_list )[0],3)
        spearmanr = round(scipy.stats.spearmanr( y_predict_list, y_test_list )[0],3)
        t = [ tf_name, cell_name,  peak_most_common, L, num_peak, len(data_train), len(data_valid), len(data_test), mse, pearsonr, spearmanr]
        results.append(t)
        print(  tf_name +' in '  + cell_name + ' is over ', t)

results = pd.DataFrame(results)
results.columns = ['tf_name', 'cell_name','peak_most_common','L','num_peak','len_data_train','len_data_valid','len_data_test','mse','pearsonr','spearmanr']
results.to_csv('data_stats_deepTFA_RNN.txt', sep='\t', header=True, index=None)

stop_time_script = time.time()
times_seconds, times_minutes, times_hours = get_running_times(start_time_script, stop_time_script)
print(('predict running times {0} seconds {1} minutes and {2} hours!').format(times_seconds, times_minutes, times_hours))
