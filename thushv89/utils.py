
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def do_tsne_grouped_by_C(x,y,fig_id,title):
    x_tsne = TSNE(random_state=123123).fit_transform(x)
    x_tsne_0 = x_tsne[np.where(y==0)[0],:]
    x_tsne_1 = x_tsne[np.where(y==1)[0],:]
    palette = ['r','b']

    # We create a scatter plot.
    f = plt.figure(fig_id)
    ax2 = plt.subplot(111)
    ax2.scatter(x_tsne_0[:,0], x_tsne_0[:,1], lw=0, s=40, c='b',label='c=0')
    ax2.scatter(x_tsne_1[:,0], x_tsne_1[:,1], lw=0, s=40, c='r',label='c=1')
    ax2.set_title(title)
    plt.legend()
    plt.show()