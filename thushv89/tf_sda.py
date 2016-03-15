import tensorflow as tf
import numpy as np
import os

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

def cumsum(xs):
    values = tf.unpack(xs)
    out = []
    prev = tf.zeros_like(values[0])
    for val in values:
        s = prev + val
        out.append(s)
        prev = s
    result = tf.pack(out)
    return result

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


    def start_session(self,sess):
        init = tf.initialize_all_variables()
        sess.run(init)
        print "Session Started ...\n"

    def test_hid_output(self,sess,batch_size,inp,b_i):
        batch_xs = inp[b_i*batch_size:(b_i+1)*batch_size]
        out = self.da_layers[0].output(self.x_sym)
        print(sess.run(out,feed_dict={self.x_sym:batch_xs}))


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
        finetune_cost = None

        prediction = chained_out(self.layers,self.x_sym,len(self.layers)-1)
        exp = tf.reverse(tf.exp(prediction,name='exp_prediction'),dims=[True])
        partial_sum = tf.reverse(cumsum(exp),dims=[True])  + 1 # get the reversed partial cumulative sum
        log_at_risk = tf.log(partial_sum[at_risk])
        diff = prediction - log_at_risk
        cost = tf.reduce_sum(tf.matmul(self.y_sym, diff))

    def pretrain(self,sess,learning_rate,batch_size,iterations,x):
        from math import ceil
        n_batches = int(ceil(x.shape[0]/batch_size))
        for step in range(iterations):
            for b_i in range(n_batches):
                for l_i,l in enumerate(self.da_layers):
                    layer_pretrain_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.pre_train_costs[l_i])
                    batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
                    sess.run(layer_pretrain_step, feed_dict={self.x_sym: batch_xs})
            print('Layer-wise pre-training Iteration ',step,' finished ...')

        print('Full pre-training starting ...')
        for step in range(iterations):
            for b_i in range(n_batches):
                full_pretrain_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.pre_train_full_cost)
                batch_xs = x[b_i*batch_size:(b_i+1)*batch_size]
                sess.run(full_pretrain_step, feed_dict={self.x_sym: batch_xs})
            print('Full pre-traning Iteration ',str(step),': ',sess.run(self.pre_train_full_cost,feed_dict={self.x_sym: batch_xs}))

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

from  sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    b_x,b_c,b_t = load_data_mat('..'+os.sep+'data'+os.sep+'Brain_P.mat')
    mmscaler = MinMaxScaler()
    norm_b_x = mmscaler.fit_transform(b_x)

    batch_size = 100
    iterations = 50

    sda = SDA_TF(183,[250,250],1,batch_size=batch_size)
    sess = tf.Session()
    sda.start_session(sess)
    sda.build_costs()

    sda.pretrain(sess,0.05,batch_size,iterations,norm_b_x)

