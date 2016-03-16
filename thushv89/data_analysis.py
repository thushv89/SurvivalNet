import numpy as np
import os
def load_data(filename):
    import pickle

    brain_p = 'Brain_P.pickle'
    brain_p_data = pickle.load( open(filename+os.sep+brain_p, "rb" ))
    print brain_p,' size: ', len(brain_p_data)
    print brain_p,' unique: ', np.unique(brain_p_data)

    luad_p = 'LUAD_P.pickle'
    luad_p_data = pickle.load( open(filename+os.sep+luad_p, "rb" ))
    print luad_p,' size: ', len(luad_p_data)
    print luad_p,' unique: ', np.unique(luad_p_data)

    lusc_p = 'LUSC_P.pickle'
    lusc_p_data = pickle.load( open(filename+os.sep+lusc_p, "rb" ))
    print lusc_p,' size: ', len(lusc_p_data)
    print lusc_p,' unique: ', np.unique(lusc_p_data)

    va = 'VA.pickle'
    va_data = pickle.load( open(filename+os.sep+va, "rb" ))
    print va,' size: ', len(va_data)
    print va,' unique: ', np.unique(va_data)

def load_data_mat(filename):
    import scipy.io as sio
    brain_data = sio.loadmat(filename)
    print '\n================= ',filename,' ================='
    for k,v in brain_data.items():

        if isinstance(v,basestring):
            print k,', Value (String): ',v
        elif isinstance(v,list):
            print k,', Size (list): ',len(v)
        elif isinstance(v,np.ndarray):
            print k,', Size (numpy): ',v.shape,' Min,Max: ',np.min(v),',',np.max(v)
        else:
            raise NotImplementedError
    print '================================================\n'
    return brain_data['X'],brain_data['C'],brain_data['T']
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

if __name__=='__main__':

    #load_data('../data/')
    b_x,b_c,b_t = load_data_mat('..'+os.sep+'data'+os.sep+'Brain_P.mat')
    #ld_x,ld_c,ld_t = load_data_mat('..'+os.sep+'data'+os.sep+'LUAD_P.mat')
    #lc_x, lc_c, lc_t = load_data_mat('..'+os.sep+'data'+os.sep+'LUSC_P.mat')
    #v_x,v_c,v_t = load_data_mat('..'+os.sep+'data'+os.sep+'VA.mat')

    # plot T against time
    index = np.arange(0,b_x.shape[0],1)
    fig = plt.figure(1)
    ax1_1 = plt.subplot(231)
    ax1_1.scatter(index,b_t)
    ax1_1.set_title('T against time')

    # plot ordered T
    index = np.arange(0,b_x.shape[0],1)
    ax1_2 = plt.subplot(232)
    ord_b_t = np.sort(b_t[:,0])
    ax1_2.scatter(index,ord_b_t)
    ax1_2.set_title('T ordered by value')

    # plot T differentiated by C
    b_t_0 = b_t[np.where(b_c==0)]
    b_t_1 = b_t[np.where(b_c==1)]
    index_0 = np.arange(0,b_t_0.size)
    index_1 = np.arange(0,b_t_1.size)
    ax1_3 = plt.subplot(233)
    ax1_3.scatter(index_0, b_t_0, c='b',label='c=0')
    ax1_3.scatter(index_1, b_t_1, c='r',label='c=1')
    ax1_3.set_title('T grouped by C')
    plt.legend()

    sorted_T = np.sort(b_t[:,0]).tolist()
    at_risk = np.asarray([sorted_T.index(x)+1 for x in sorted_T]).astype('int32')
    index = np.arange(0,b_x.shape[0],1)
    ax1_4 = plt.subplot(234)
    ax1_4.scatter(index,np.asarray(at_risk)-1)
    ax1_4.set_title('at risk - 1')

    args_sorted_T = np.argsort(b_t[:,0]).tolist()
    C_sorted_by_T = b_c[args_sorted_T]
    index = np.arange(0,b_x.shape[0],1)
    ax1_5 = plt.subplot(235)
    ax1_5.scatter(index,C_sorted_by_T)
    ax1_5.set_title('C sorted by T')

    index = np.arange(0,b_x.shape[0],1)
    ax1_6 = plt.subplot(236)
    ax1_6.scatter(index,b_c)
    ax1_6.set_title('C against time')

    plt.show()
    # plot tsne transform of X
    b_x_tsne = TSNE(random_state=123123).fit_transform(b_x)
    b_x_tsne_0 = b_x_tsne[np.where(b_c==0)[0],:]
    b_x_tsne_1 = b_x_tsne[np.where(b_c==1)[0],:]
    palette = ['r','b']

    # We create a scatter plot.
    f = plt.figure(4)
    ax2 = plt.subplot(111)
    ax2.scatter(b_x_tsne_0[:,0], b_x_tsne_0[:,1], c='b',label='c=0')
    ax2.scatter(b_x_tsne_1[:,0], b_x_tsne_1[:,1], c='r',label='c=1')
    plt.legend()
    plt.show()