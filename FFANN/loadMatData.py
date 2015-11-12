import scipy.io as sio
from path import Path
import numpy as np

VA = Path('data/VA.mat')
LUAD_P = Path('data/LUAD_P.mat')
LUSC_P = Path('data/LUSC_P.mat')
Brain_P = Path('data/Brain_P.mat')
LUAD_G = Path('data/LUAD_G.mat')
LUSC_G = Path('data/LUSC_G.mat')


def discrete_time_data(old_x, observed, survival_time, start=0.1, sort=False):
    if sort:
        order = np.argsort(survival_time)
        old_x = old_x[order]
        survival_time = survival_time[order]
        observed = observed[order]
    x = []
    new_observed = []
    # each entry in x is a list of all time prior than x
    for index in xrange(len(old_x)):
        data = old_x[index]
        temp = list(data)
        time = survival_time[index]
        step = start
        while step < time:
            new_row = temp[:]
            new_row[0] = step
            x.append(new_row)
            new_observed.append(0.0)
            step += start
        temp[0] = step
        x.append(temp)
        if observed[index]:
            new_observed.append(1.0)
        else:
            new_observed.append(0.0)
    return np.asarray(x), np.asarray(new_observed)


def load_data(p=Brain_P, step=0.1):
    print "loading data..."
    observed, X, survival_time = load_mat_data(p)
    test_size = len(X) / 3
    train_X = X[test_size:]
    # print train_X.shape
    train_y = survival_time[test_size:]
    train_X, train_observed = discrete_time_data(train_X, observed, train_y, start=step)
    test_X = X[:test_size]
    test_observed = observed[:test_size]
    test_y = survival_time[:test_size]
    print train_X.shape
    return train_X, train_observed, test_y, test_observed, test_X


def load_mat_data(p, sort=True):
    mat = sio.loadmat(p)
    X = mat['X']
    C = mat['C']
    T = mat['T']
    if sort:
        survival_time = np.asarray([t[0] for t in T])
        order = np.argsort(survival_time)
        censored = np.asarray([c[0] for c in C], dtype='int32')
        # print survival_time
        return 1 - censored[order], X[order].astype(float), survival_time[order]
    else:
        survival_time = np.asarray([t[0] for t in T])
        censored = np.asarray([c[0] for c in C], dtype='int32')
        return 1 - censored, X.astype(float), survival_time


def save_csv(name="LUAD_P.csv", p=LUAD_P):
    observed, X, survival_time = load_mat_data(p=p)
    print X.shape
    X = np.insert(X, 0, observed, axis=1)
    X = np.insert(X, 0, survival_time, axis=1)
    print X.shape
    np.savetxt(Path('C:/Users/Song/Research/biomed/Survival/RSF/' + name), X, delimiter=',', fmt='%10.5f')

if __name__ == '__main__':
    save_csv(name="LUAD_G.csv", p=LUAD_G)
    save_csv(name="LUSC_G.csv", p=LUSC_G)
