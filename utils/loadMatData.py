__author__ = 'Song'
import scipy.io as sio
import matlab.engine
import numpy as np
import cPickle

#
#VA = Path('data/VA.mat')
#LUAD_P = Path('data/LUAD_P.mat')
#LUSC_P = Path('data/LUSC_P.mat')
#Brain_P = Path('data/Brain_P.mat')


#def save_pickle(name='LUAD_P.pickle', p=LUAD_P):
#    observed, x, y, at_risk = load_data(p=p)
#    f = file(name, 'wb')
#    for obj in [observed, x, y, at_risk]:
#        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
#    f.close()


def read_pickle(name='./datasets/LUAD_P.pickle'):
    f = file(name, 'rb')
    loaded_objects = []
    for i in range(4):
        loaded_objects.append(cPickle.load(f))
    observed, X, survival_time, at_risk_X = loaded_objects
    f.close()
    print at_risk_X
    return observed, X, survival_time, at_risk_X


#def load_data(p=LUAD_P):
#    mat = sio.loadmat(p)
#    X = mat['X']
#    C = mat['C']
#    T = mat['T']
#    eng = matlab.engine.start_matlab()
#    survival_time = [t[0] for t in T]
#    mat_T = matlab.double(survival_time)
#    survival_time, order = eng.sort(mat_T, nargout=2)
#    order = np.asarray(order[0]).astype(int) - 1
#    input_len = len(order) / 3
#    temp = matlab.double(survival_time[0][input_len:])
#    at_risk = np.asarray(eng.ismember(temp, temp, nargout=2)[1][0]).astype(int)
#    censored = np.asarray([c[0] for c in C], dtype='int32')
#    survival_time = np.asarray(survival_time[0])
#    # print survival_time
#    return 1 - censored[order], X[order], survival_time, at_risk - 1

if __name__ == '__main__':
    # save_pickle(name='VA.pickle', p=VA)
    # save_pickle(name='LUSC_P.pickle', p=LUSC_P)
    # save_pickle(name='Brain_P.pickle', p=Brain_P)
    read_pickle(name='./datasets/VA.pickle')