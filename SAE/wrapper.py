# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:35:38 2015

@author: syouse3
"""
import os
import time
from train import test_SdA

def wrapper():
    pathout = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct22/'
    
    #test_SdA(finetune_lr=0.01, pretraining_epochs=40, n_layers=3, n_hidden=140,
    #         pretrain_lr=1.0, training_epochs=200, batch_size=2, augment=False,
    #         drop_out=True, pretrain_dropout=False, dropout_rate=0.3)
    if not os.path.exists(pathout):
        os.makedirs(pathout)
        
    layers = [2]
    hSizes = [20]
    #do_rates = [.7, .5, .3, .1, 0]
    do_rates = [0]
    for hSize in hSizes:
        for n_layer in layers:
            for do_rate in do_rates:
                t = time.time()
                test_SdA(finetune_lr=0.001, pretraining_epochs=200, n_layers=n_layer,\
                n_hidden = hSize, pretrain_lr=1.0, training_epochs=200, batch_size=100, \
                augment = False, drop_out = True, pretrain_dropout=False, dropout_rate= do_rate, \
                resultPath = pathout)
                elapsed = time.time() - t
                print 'Elapsed: %s' %elapsed                
if __name__ == '__main__':
    wrapper()