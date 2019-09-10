from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import seaborn as sns
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
Categorical = tf.contrib.distributions.Categorical
Dirichlet = tf.contrib.distributions.Dirichlet

#%%
def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def categorical_loglikelihood(b, z_concate):
    '''
    b is N*K*n_cv*n_class, one-hot vector in row
    z_concate is logits, softplus(z_concate) is prob
    z_concate is N*1*n_cv*n_class, first column 0
    '''
    lik_v = b*(z_concate-tf.reduce_logsumexp(z_concate,axis=3,keep_dims=True))
    return tf.reduce_sum(lik_v,axis=3)
 
def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-5
    q = tf.nn.softmax(q_logit,dim=2)
    p = tf.nn.softmax(p_logit,dim=2)
    return tf.reduce_sum(q*(tf.log(q+eps)-tf.log(p+eps)),axis = [1,2])
       

def encoder(x,z_dim,reuse=False):
    #return logits [N,K,n_cv*(n_class-1)]
    #z_dim is n_cv*(n_class-1)
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        #h1 = tf.layers.dense(2. * x - 1., 200, tf.nn.relu, name="encoder_1")
        #h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
        z = tf.layers.dense(h2, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim,reuse=False):
    #return logits
    #b is [N,K,n_cv,n_class]
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        shape = b.get_shape().as_list()
        latent_dim = np.prod(shape[2:]) #equal to z_concate_dim
        b = tf.reshape(b, [-1, shape[1],latent_dim])    
        h2 = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        #h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
        #h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
        logit_x = tf.layers.dense(h2, x_dim, activation = None)
    return logit_x
    



def fun(x_star,E,prior_logit0,z_concate,axis_dim=2,reuse_decoder=False):
    '''
    x_star is N*K*d_x, E is N*K*n_cv*n_class, z_concate is N*n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    '''
    #KL
    #logits_py = tf.ones_like(z_concate) #uniform
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    logits_py = tf.tile(prior_logit1,[tf.shape(E)[0],1,1])
    
    #p_cat_b = Categorical(logits=logits_py+1e-4)
    #q_cat_b = Categorical(logits=z_concate+1e-4)
    #KL_qp = tf.contrib.distributions.kl_divergence(q_cat_b, p_cat_b)
    #KL = tf.reduce_sum(KL_qp,1)
    
    KL = kl_cat(z_concate, logits_py)
    
    #log p(x_star|E)
    logit_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, logit_x)
    # (N,K)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    return - log_p_x_given_b + KL
    

#%%
EXPERIMENT = 'rfc_1'
directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
batch_size = 200  
training_epochs = 1000
np_lr = 0.0001 
N_montecarlo = 1

    
#%%
tf.reset_default_graph() 

x_dim = 784
n_class = 10 ; n_cv = 20  # # of classes and # of cat variables
z_dim = n_cv * (n_class-1)   # # of latent parameters neede for one cat var is n_cat-1
z_concate_dim = n_cv * n_class

eps = 1e-10
K_u = 1; K_b = 1
lr=tf.constant(0.0001)

prior_logit0 = tf.zeros([n_cv,n_class])

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#encoder q(b|x) = log Cat(b|z_concate)
#logits for categorical, p=softmax(logits)
z0 = encoder(x_binary,z_dim)  #N*d_z
z = tf.reshape(z0,[N,n_cv,n_class-1])
zeros_logits = tf.zeros(shape = [N,n_cv,1])
z_concate = tf.concat([zeros_logits,z],axis=2) #N*n_cv*n_class

prob = tf.nn.softmax(z_concate,dim=2)

q_b = Categorical(logits=z_concate) #sample K_b \bv
#non-binary, accompanying with encoder parameter, cannot backprop
b_sample0 = q_b.sample(K_b) #K_b*N*n_cv
b_sample = tf.one_hot(tf.transpose(b_sample0,perm=[1,0,2]),depth=n_class)  #N*K_b*n_cv
b_sample = tf.cast(b_sample,tf.float32)

#compute decoder p(x|b), gradient of encoder parameter can be automatically given by loss
x_star_b = tf.tile(tf.expand_dims(x_binary,axis=1),[1,K_b,1]) #N*K_b*d_x
#average over K_b
gen_loss0 = tf.reduce_mean(fun(x_star_b,b_sample,prior_logit0,z_concate,reuse_decoder= False),axis=1) 
gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)


#provide encoder q(b|x) gradient


alpha_grads = tf.placeholder(tf.float32,[None,z_dim]) 
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is #N*z_dim, alpha_grads is N*z_dim, inf_vars is d_theta
#d_theta, should devide by batch-size, but can be absorb into learning rate
inf_grads = tf.gradients(z0, inf_vars, grad_ys=alpha_grads)#/b_s
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)

#prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])


with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% TRAIN
# get data
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation


total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)

display_step = total_batch


def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)
        

def compute_arm(train_xs, K=1):
    grad_phi = 0.
    for i in range(K):
        F = sess.run(gen_loss0, {x:train_xs}) #N,
        F = F[:,None]
        P = sess.run(z_concate, {x:train_xs}) #N*20*10
        Cat = np.squeeze(sess.run(b_sample, {x:train_xs})) #N*20*10
        grad_phi_i = np.reshape((Cat-P)[:,:,1:],[batch_size,-1])*F #N*180
        grad_phi += grad_phi_i
    grad_phi = grad_phi/K
    return grad_phi

record = [];step = 0
import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]


print('Training starts....',EXPERIMENT)
sess=tf.InteractiveSession()
sess.run(init_op)



for epoch in range(training_epochs):
    avg_cost = 0.
    avg_cost_test = 0.
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size) 
        grad_phi = compute_arm(train_xs,N_montecarlo)
        _,cost = sess.run([gen_train_op,gen_loss],{x:train_xs,alpha_grads:grad_phi, lr:np_lr})
        record.append(cost)
        step += 1
    
        
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch)
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(valid_loss)
        print(epoch,'valid_cost=',valid_loss,'with std=',np.std(record))

    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'w'))
    record=[]

print(EXPERIMENT)

























