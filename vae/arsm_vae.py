#ARSM
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle
import argparse
from keras.utils.np_utils import to_categorical 
import warnings
warnings.filterwarnings('ignore')

slim=tf.contrib.slim
Categorical = tf.contrib.distributions.Categorical
Dirichlet = tf.contrib.distributions.Dirichlet
Flatten = tf.keras.layers.Flatten()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', '-l', type=float, default=0.0003, help='lr')
parser.add_argument('--name', '-n', default='arsm', help='model name')
parser.add_argument('--n_cv', type=int, default=20, help='number of cat var')
parser.add_argument('--n_class', type=int, default=10, help='number of class')
parser.add_argument('--batch', '-b', type=int, default=200, help='mini-batch size')
parser.add_argument('--epoch', '-e', type=int, default=1000, help='number of epoch')

args = parser.parse_args()

#%%

def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def bernoulli_loglikelihood(b, logits):
    '''
    input: N*d; output: N*d 
    '''
    return b * (-tf.nn.softplus(-logits)) + (1 - b) * (-logits - tf.nn.softplus(-logits))

def categorical_loglikelihood(b, logits):
    '''
    b is N*n_cv*n_class, one-hot vector in row
    logits is N*n_cv*n_class, softmax(logits) is prob
    return: N*n_cv
    '''
    lik_v = b*(logits-tf.reduce_logsumexp(logits,axis=-1,keep_dims=True))
    return tf.reduce_sum(lik_v,axis=-1)
        

def encoder(x,z_dim):
    '''
    return logits [N,n_cv*(n_class-1)]
    z_dim is n_cv*(n_class-1)
    '''
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        h = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        z = tf.layers.dense(h, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim):
    '''
    return logits
    b is [N,n_cv,n_class]
    '''
    with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
        b = Flatten(b)
        h = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        logit_x = tf.layers.dense(h, x_dim, activation = None)
    return logit_x


def fun(x_binary,E,prior_logit0,z_concate):
    '''
    x_binary is N*d_x, E is N*n_cv*n_class, z_concate is N*n_cv*n_class
    prior_logit0 is n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    return (N,)
    '''
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    logits_py = tf.tile(prior_logit1,[tf.shape(E)[0],1,1]) 
    #log p(x|z)
    logit_x = decoder(E,x_dim)
    log_p_x_given_z = tf.reduce_sum(bernoulli_loglikelihood(x_binary, logit_x), axis=1) 
    #log q(z|x)
    log_q_z_given_x = tf.reduce_sum(categorical_loglikelihood(E, z_concate), axis=1)    
    #log p(z)
    log_p_z = tf.reduce_sum(categorical_loglikelihood(E, logits_py), axis=1)
    
    return - log_p_x_given_z - log_p_z + log_q_z_given_x
    

def Fn(pai,prior_logit0,z_concate,x_star_u):
    '''
    pai is [N,n_cv,n_class]
    z_concate is [N,n_class]
    '''
    z_concate1 = tf.expand_dims(z_concate,axis=1)
    E = tf.one_hot(tf.argmin(tf.log(pai+eps)-z_concate1,axis = 3),depth=n_class)
    E = tf.cast(E,tf.float32)
    return fun(x_star_u,E,prior_logit0,z_concate)
     
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)

def compt_F(sess, dirich, logits, xs):
    FF = np.zeros([batch_size, n_class, n_class])  
    for i in range(n_class):
        for j in range(i,n_class):
            dirich_ij = np.copy(dirich)
            dirich_ij[:,:,[i,j]] = dirich_ij[:,:,[j,i]]
            s_ij  = to_categorical(np.argmin(np.log(dirich_ij+eps)-logits, axis = -1),num_classes=n_class)
            FF[:,i,j] = sess.run(F_ij,{x:xs, EE:s_ij})
            FF[:,j,i] = FF[:,i,j]
    return FF

    
#%% Model
    
tf.reset_default_graph() 

x_dim = 784
n_class = args.n_class ; n_cv = args.n_cv  
z_dim = n_cv * (n_class-1)  
z_concate_dim = n_cv * n_class

eps = 1e-10
lr = args.lr

prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([n_cv,n_class]))

x = tf.placeholder(tf.float32,[None,x_dim]) 
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#encoder q(z|x)
z0 = encoder(x_binary,z_dim) 
z = tf.reshape(z0,[N,n_cv,n_class-1])
zeros_logits = tf.zeros(shape = [N,n_cv,1])
z_concate = tf.concat([zeros_logits,z],axis=2)
q_b = Categorical(logits=z_concate)

b_sample = q_b.sample() 
b_sample = tf.one_hot(b_sample,depth=n_class) 
b_sample = tf.cast(b_sample,tf.float32)

#compute decoder p(x|z) gradient 
gen_loss0 = fun(x_binary,b_sample,prior_logit0,z_concate)
gen_loss = tf.reduce_mean(gen_loss0)
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

#compute encoder q(z|x) gradient 
Dir = Dirichlet([1.0]*n_class)
pai = Dir.sample(sample_shape=[N,n_cv]) 

EE = tf.placeholder(tf.float32,[None, n_cv, n_class]) 
F_ij = fun(x_binary,EE,prior_logit0,z_concate)
      
F = tf.placeholder(tf.float32,[None,n_class,n_class]) #symmetric
tilde_F = F - tf.reduce_mean(F, axis = -1, keep_dims=True)
PAI = tf.placeholder(tf.float32,[None,n_cv,n_class])
tilde_pi = 1/n_class - PAI
alpha_grads0 = tf.matmul(tilde_pi, tilde_F)
alpha_grads = tf.reshape(alpha_grads0[:,:,1:],[-1,z_dim])
        
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

inf_grads = tf.gradients(z0, inf_vars, grad_ys=alpha_grads)
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)

prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])

with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% data

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
batch_size = args.batch 
training_epochs = args.epoch

mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

total_batch = int(mnist.train.num_examples / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)


#%% TRAIN
        
print('Training starts....',args.name)

sess=tf.InteractiveSession()
sess.run(init_op)
record = [];step = 0

COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[]
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_cost_test = 0.
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size) 
        dirich, logits = sess.run([pai, z_concate],{x:train_xs})
        FF = compt_F(sess, dirich, logits, train_xs)                
        _,cost = sess.run([train_op,gen_loss],{x:train_xs, F:FF, PAI:dirich})
        record.append(cost)
        step += 1
        
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch)
        COUNT.append(step); COST.append(np.mean(record)); 
        COST_VALID.append(valid_loss)
        print(epoch,'train_cost=',np.mean(record),'valid_cost=',valid_loss)

    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list]
        cPickle.dump(all_, open(directory+args.name, 'w'))
    record=[]



