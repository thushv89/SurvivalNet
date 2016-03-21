import tensorflow as tf
import numpy as np
import os
from lifelines.utils import _naive_concordance_index

class Layer_TF(object):

    def __init__(self,n_ins,n_outs):
        self.W = tf.Variable(tf.random_uniform((n_ins,n_outs),
                                   minval=-4 * np.sqrt(6. / (n_ins + n_outs)),
                                   maxval=4 * np.sqrt(6. / (n_ins + n_outs)),
                                   dtype=tf.float32, seed=None, name=None),name='W'+str(n_ins)+'-'+str(n_outs))
        self.b = tf.Variable(tf.random_uniform((n_outs,), minval=0, maxval=0.1,
                                               dtype=tf.float32, seed=None,
                                               name='b'+str(n_ins)+'-'+str(n_outs)))
        self.b_prime = tf.Variable(tf.random_uniform((n_ins,), minval=0, maxval=0.1,
                                                     dtype=tf.float32, seed=None,
                                                     name='b_prime'+str(n_ins)+'-'+str(n_outs)))
        self.out = None
        self.rec = None

    def output(self,x):
        self.out = tf.nn.sigmoid(tf.matmul(x,self.W)+self.b,name='sigmoid_out')
        return self.out

    def reconstruct(self,out):
        self.rec = tf.nn.sigmoid(tf.matmul(out,tf.transpose(self.W,name='transpose_W'))+self.b_prime)
        return self.rec

def data_ordered_by_risk(X, T, C):
    tmp = list(T)
    T = np.asarray(tmp).astype('float64')
    order = np.argsort(T)
    sorted_T = T[order]
    at_risk = np.asarray([list(sorted_T).index(x)+1 for x in sorted_T]).astype('int32').reshape(-1,1)
    T = np.asarray(sorted_T).reshape(-1,1)
    C = C[order]
    X = X[order]
    return X, T, C, at_risk - 1

def chained_out(layers,x,hid_idx):
    out = x
    for l in layers[:hid_idx+1]:
        out = l.output(out)
    return out

def chained_rec(layers,out,hid_idx):
    inp = out
    for l in reversed(layers[:hid_idx+1]):
        inp = l.reconstruct(inp)
    return inp

class SDA_TF(object):

    def __init__(self,
        n_ins=183,
        hidden_layers_sizes=[500, 500],
        n_outs=1,
        batch_size=10,
        corruption_levels=[0.1, 0.1],
        at_risk=None,
        at_risk_test=None,
        drop_out=False,
        pretrain_dropout=False,
        dropout_rate=0.1,
        non_lin=None,
        alpha=None):

        self.n_ins = n_ins
        self.n_outs = n_outs
        self.n_hid_list = hidden_layers_sizes
        self.corruption_levels = corruption_levels
        self.batch_size = batch_size

        n_all_layers = []
        n_all_layers.append(self.n_ins)
        n_all_layers.extend(self.n_hid_list)
        n_all_layers.append(self.n_outs)

        self.layers = []
        for l_in,l_out in zip(n_all_layers[:-1],n_all_layers[1:]):
            self.layers.append(Layer_TF(l_in,l_out))
        self.da_layers = self.layers[:-1]
        self.log_layer = self.layers[-1]

        self.pre_train_costs = []
        self.pre_train_full_cost = None
        self.finetune_cost = None

        self.x_sym = tf.placeholder(tf.float32, [None, self.n_ins],name='x') # None means that dimension could be anythin
        self.y_sym = tf.placeholder(tf.float32, [None, self.n_outs],name='y')
        self.risk_sym = tf.placeholder(tf.int32, [None, 1],name='risk') # place holder for at-risk variable
        self.psum_sym = tf.placeholder(tf.float32, [None, 1],name='partial_sum') # place holder for at-risk variable
        ###for testing
        #self.exp = None

    def start_session(self,sess):
        init = tf.initialize_all_variables()
        sess.run(init)
        print "Session Started ...\n"

    def hid_output(self,sess,batch_size,x,layer_idx):
        from math import ceil
        n_batches = int(ceil(x.shape[0]*1.0/batch_size))

        hid_features = None
        for b_i in range(n_batches):
            batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
            out = chained_out(self.da_layers,self.x_sym,layer_idx)
            if hid_features is None:
                hid_features = sess.run(out,feed_dict={self.x_sym:batch_xs})
            else:
                hid_features = np.append(hid_features,sess.run(out,feed_dict={self.x_sym:batch_xs}),axis=0)

        return hid_features


    def build_costs(self):

        # layer-wise pre-training
        inp = self.x_sym
        for l_i,l in enumerate(self.da_layers):
            out = l.output(inp)
            inp_hat = l.reconstruct(out)
            l_cost = tf.reduce_mean(inp*tf.log(inp_hat), name='xentropy_mean_layer_'+str(l_i))
            self.pre_train_costs.append(l_cost)
            inp = out

        # full pre-training
        last_out = chained_out(self.da_layers,self.x_sym,len(self.da_layers)-1)
        x_hat = chained_rec(self.da_layers,last_out,len(self.da_layers)-1)
        self.pre_train_full_cost = -tf.reduce_mean(self.x_sym*tf.log(x_hat), name='xentropy_mean_full')

        # to be implemented
        prediction = chained_out(self.layers,self.x_sym,len(self.layers)-1)
        log_at_risk = tf.log(self.psum_sym,name='log_partial_sum')
        diff = prediction - log_at_risk
        #self.finetune_cost = -tf.reduce_mean(tf.matmul(tf.transpose(self.y_sym), diff))
        self.finetune_cost = -tf.reduce_mean(self.y_sym* diff)

    def pretrain(self,sess,learning_rate, batch_size, iterations, x):
        from math import ceil
        n_batches = int(ceil(x.shape[0]*1.0/batch_size))
        for step in range(iterations):
            for b_i in range(n_batches):
                for l_i,l in enumerate(self.da_layers):
                    layer_pretrain_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.pre_train_costs[l_i])
                    batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
                    sess.run(layer_pretrain_step, feed_dict={self.x_sym: batch_xs})
            print 'Layer-wise pre-training Iteration ',step,' finished ...'

        print 'Full pre-training starting ...'
        for step in range(iterations):
            for b_i in range(n_batches):
                full_pretrain_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.pre_train_full_cost)
                batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
                sess.run(full_pretrain_step, feed_dict={self.x_sym: batch_xs})
            print 'Full pre-traning Iteration ',str(step),': ',sess.run(self.pre_train_full_cost,feed_dict={self.x_sym: batch_xs})

    def finetune(self,sess,learning_rate,batch_size,iterations,x,y,at_risk):
        print '\n Fine tuning... \n'
        from math import ceil
        n_batches = int(ceil(x.shape[0]*1.0/batch_size))
        for step in range(iterations):
            for b_i in range(n_batches):

                batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
                batch_ys = y[b_i*batch_size:(b_i+1)*batch_size]
                batch_risk = at_risk[b_i*batch_size:(b_i+1)*batch_size]

                prediction = sess.run(chained_out(self.layers,self.x_sym,len(self.layers)-1),feed_dict={self.x_sym:x})
                exp = np.exp(prediction)[::-1]
                partial_sum = np.cumsum(exp)[::-1] + 1 # get the reversed partial cumulative sum
                batch_psum = np.log(partial_sum[batch_risk])

                finetune_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.finetune_cost)


                sess.run(finetune_step, feed_dict={self.x_sym: batch_xs,
                                                   self.y_sym:batch_ys,
                                                   self.risk_sym:batch_risk,
                                                   self.psum_sym:batch_psum}
                         )

            print 'Fine-tuning Iteration ',str(step),': ', \
                sess.run(self.finetune_cost,feed_dict={self.x_sym: batch_xs,self.y_sym:batch_ys,self.risk_sym:batch_risk,self.psum_sym:batch_psum})

    def calc_CI(self,sess, test_x,test_C,test_T):
        print '\n Calculating Concordance Index... \n'
        from math import ceil
        c = []
        n_batches = int(ceil(test_x.shape[0]*1.0/batch_size))

        hazard = None
        for b_i in range(n_batches):
            batch_xs = test_x[b_i*batch_size:(b_i+1)*batch_size]
            pred = sess.run(chained_out(self.layers,self.x_sym,len(self.layers)-1),feed_dict={self.x_sym:batch_xs})
            if hazard is None:
                hazard = pred
            else:
                hazard = np.append(hazard,pred,axis=0)

        c_index = _naive_concordance_index(test_T, -hazard, test_C)
        return c_index

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
    return brain_data['X'],1-brain_data['C'].flatten(),brain_data['T'].flatten()

from  sklearn.preprocessing import MinMaxScaler
from utils import do_tsne_grouped_by_C
if __name__ == '__main__':

    quick_test_mode = False
    tsne = False

    if quick_test_mode:
        print "################## WARNING: Quick Test Mode #################"

    b_x,b_c,b_t = load_data_mat('..'+os.sep+'data'+os.sep+'Brain_P.mat')
    mmscaler = MinMaxScaler()
    norm_b_x = mmscaler.fit_transform(b_x)

    batch_size = 1
    iterations = 20
    hid_sizes = [256,128]
    if quick_test_mode:
        batch_size = 100
        iterations = 2
        hid_sizes = [32,32]

    pt_learning_rate = 0.04
    ft_learning_rate = 0.1

    fold_size = 10
    data_per_fold = int(norm_b_x.shape[0]*1.0/fold_size)

    sda = SDA_TF(183,hid_sizes,1,batch_size=batch_size)
    sess = tf.Session()
    sda.start_session(sess)
    sda.build_costs()

    for fold in range(fold_size):
        print '\n======================= Fold ',fold,'=======================\n'
        X_test = norm_b_x[fold*data_per_fold:(fold+1)*data_per_fold]
        T_test = b_t[fold*data_per_fold:(fold+1)*data_per_fold]
        C_test = b_c[fold*data_per_fold:(fold+1)*data_per_fold]

        X_train = np.delete(norm_b_x,range(fold*data_per_fold,(fold+1)*data_per_fold),axis=0)
        T_train = np.delete(b_t,range(fold*data_per_fold,(fold+1)*data_per_fold),axis=0)
        C_train = np.delete(b_c,range(fold*data_per_fold,(fold+1)*data_per_fold),axis=0)

        ord_b_x_train,ord_b_c_train,ord_b_t_train,at_risk_train = data_ordered_by_risk(X_train,T_train,C_train)
        ord_b_x_test,ord_b_c_test,ord_b_t_test,at_risk_test = data_ordered_by_risk(X_test,T_test,C_test)

        sda.pretrain(sess,pt_learning_rate,batch_size,iterations,X_train)

        if tsne:
            features_0 = sda.hid_output(sess,batch_size,norm_b_x,0)
            do_tsne_grouped_by_C(features_0,b_c,1,'TSNE visualization for features in hidden layer 0')

            if len(hid_sizes)>1:
                features_1 = sda.hid_output(sess,batch_size,norm_b_x,1)
                do_tsne_grouped_by_C(features_1,b_c,2,'TSNE visualization for features in hidden layer 1')

        sda.finetune(sess,ft_learning_rate,batch_size,iterations,ord_b_x_train,ord_b_c_train,at_risk_train)
        ci_index = sda.calc_CI(sess,ord_b_x_test,ord_b_c_test,ord_b_t_test)
        print 'Concordance Index for Fold', fold,': ',ci_index

