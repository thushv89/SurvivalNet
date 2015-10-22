# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:53:54 2015

@author: syouse3
"""

import matplotlib.pyplot as plt
import cPickle

def plotresults():
    #measure = 'lpl'
    path = '/home/syouse3/git/Survivalnet/SAE/results/LUAD-Oct22/'
    paths = [path + 'ftlr0.001-pt200-nl2-hs60-ptlr1.0-ft200-bs100-auFalse-dor0-doFalseFalselpl',\
    path + 'ftlr0.001-pt200-nl2-hs60-ptlr1.0-ft200-bs100-auFalse-dor0-doFalseFalselpl', \
    path + 'ftlr0.001-pt200-nl6-hs60-ptlr1.0-ft200-bs100-auFalse-dor0-doFalseFalselpl', \
    path + 'ftlr0.001-pt200-nl8-hs60-ptlr1.0-ft200-bs100-auFalse-dor0-doFalseFalselpl', \
    path + 'ftlr0.001-pt200-nl10-hs60-ptlr1.0-ft200-bs100-auFalse-dor0-doFalseFalselpl']
    markers = ['o', '*', '^', '.', 'v']   
    colors = ['r', 'b', 'g', 'm', 'c']    
    for i in range(5):
        f = file(paths[i], 'rb')
        loaded_objects = []
        loaded_objects = (cPickle.load(f))
        f.close()
        plt.plot(range(len(loaded_objects)), loaded_objects, c=colors[i], marker=markers[i], lw=5, ms=10, mfc=colors[i])
        plt.legend(['2','4','6','8','10'])
        plt.ylim([-1000, 0])
    plt.show()


    
if __name__ == '__main__':
    plotresults()
