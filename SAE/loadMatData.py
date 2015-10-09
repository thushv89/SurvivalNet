__author__ = 'Song'
import scipy.io as sio
from path import Path
import matlab.engine
import numpy as np


def load_data(p=Path('data/LUAD_P.mat')):
    mat = sio.loadmat(p)
    X = mat['X']
    C = mat['C']
    T = mat['T']
    eng = matlab.engine.start_matlab()
    survival_time = [t[0] for t in T]
    mat_T = matlab.double(survival_time)
    survival_time, order = eng.sort(mat_T, nargout=2)
    order = np.asarray(order[0]).astype(int) - 1
    input_len = len(order) / 3
    temp = matlab.double(survival_time[0][input_len:])
    at_risk = np.asarray(eng.ismember(temp, temp, nargout=2)[1][0]).astype(int)
    censored = np.asarray([c[0] for c in C], dtype='int32')
    survival_time = np.asarray(survival_time[0])
    return 1 - censored[order], X[order], survival_time, at_risk - 1

if __name__ == '__main__':
    load_data()