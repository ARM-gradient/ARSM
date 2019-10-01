#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

import warnings
warnings.filterwarnings('ignore')
slim=tf.contrib.slim

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
        

def encoder(x,z_dim,reuse=False):
    #return logits [N,n_cv*(n_class-1)]
    #z_dim is n_cv*(n_class-1)
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        z = tf.layers.dense(h2, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim,reuse=False):
    #return logits
    #b is [N,K,n_cv,n_class]
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        shape = b.get_shape().as_list()
        latent_dim = np.prod(shape[1:]) #equal to z_concate_dim
        b = tf.reshape(b, [-1, latent_dim])    
        h2 = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        logit_x = tf.layers.dense(h2, x_dim, activation = None)
    return logit_x
    

def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-5
    q = tf.nn.softmax(q_logit,dim=2)
    p = tf.nn.softmax(p_logit,dim=2)
    return tf.reduce_sum(q*(tf.log(q+eps)-tf.log(p+eps)),axis = [1,2])


def fun(x_star,E,prior_logit0,z_concate,axis_dim=1,reuse_decoder=False):
    '''
    x_star is N*d_x, E is N*n_cv*n_class, z_concate is N*n_cv*n_class
    prior_logit0 is n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return (N,)
    '''
    #KL
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    logits_py = tf.tile(prior_logit1,[tf.shape(E)[0],1,1])
    KL = kl_cat(z_concate, logits_py) 
    logit_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, logit_x)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    return - log_p_x_given_b + KL
    

def Fn(pai,prior_logit0,z_concate,x_star_u):
    '''
    pai is [N,K_u,n_cv,n_class]
    z_concate is [N,K_u,n_class]
    '''
    z_concate1 = tf.expand_dims(z_concate,axis=1)
    E = tf.one_hot(tf.argmin(tf.log(pai+eps)-z_concate1,axis = 3),depth=n_class)
    E = tf.cast(E,tf.float32)
    return fun(x_star_u,E,prior_logit0,z_concate,reuse_decoder=True)
    

#%%
EXPERIMENT = 'ARSM'
directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
batch_size = 200  
training_epochs = 1000
np_lr = 0.0001 


    
#%%
tf.reset_default_graph() 

x_dim = 784
n_class = 10 ; n_cv = 20  # # of classes and # of cat variables
z_dim = n_cv * (n_class-1)   # # of latent parameters needed for one cat var is n_cat-1
z_concate_dim = n_cv * n_class

eps = 1e-10
K_u = 1; K_b = 1
lr=tf.constant(0.0001)

prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([n_cv,n_class]))

x = tf.placeholder(tf.float32,[None,x_dim]) 
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

z0 = encoder(x_binary,z_dim)  #N*d_z
z = tf.reshape(z0,[N,n_cv,n_class-1])
zeros_logits = tf.zeros(shape = [N,n_cv,1])
z_concate = tf.concat([zeros_logits,z],axis=2) 
q_b = Categorical(logits=z_concate) 
b_sample = q_b.sample() #N*n_cv
b_sample = tf.one_hot(b_sample,depth=n_class)  
b_sample = tf.cast(b_sample,tf.float32)

x_star_b = x_binary 
gen_loss0 = fun(x_star_b,b_sample,prior_logit0,z_concate,reuse_decoder= False)

gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)


#provide encoder q(b|x) gradient by data augmentation
Dir = Dirichlet([1.0]*n_class)
pai = Dir.sample(sample_shape=[N,n_cv]) 

x_star_u = x_binary #N*d_x

EE = tf.placeholder(tf.float32,[None, n_cv, n_class]) 
F_ij = fun(x_star_u,EE,prior_logit0,z_concate,reuse_decoder= True)
      
F = tf.placeholder(tf.float32,[None,n_class,n_class]) #n_class*n_class
F0 = F - tf.reduce_mean(F, axis = 2, keep_dims=True)
F1 = tf.expand_dims(F0, axis=1)
PAI = tf.placeholder(tf.float32,[None,n_cv,n_class])
pai1 = 1/n_class - tf.tile(tf.expand_dims(PAI, axis=2),[1,1,n_class,1])

alpha_grads0 = tf.reduce_mean(F1*pai1, axis = -1)
alpha_grads = tf.reshape(alpha_grads0[:,:,1:],[-1,z_dim])

        
        
alpha_grads = tf.reshape(alpha_grads,[-1,z_dim])
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

inf_grads = tf.gradients(z0, inf_vars, grad_ys=alpha_grads)
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


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



#%%
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)

def compt_F(sess, train_xs, pai, z_concate):
    pp, ph = sess.run([pai, z_concate],{x:train_xs})
    FF = np.zeros([batch_size, n_class,n_class])
    from keras.utils.np_utils import to_categorical   
    for i in range(n_class):
        for j in range(i,n_class):
            pp_ij = np.copy(pp)
            pp_ij[:,:,[i,j]] = pp_ij[:,:,[j,i]]
            s_ij  = to_categorical(np.argmin(np.log(pp_ij+1e-6)-ph, axis = 2),num_classes=n_class)
            FF[:,i,j] = sess.run(F_ij,{x:train_xs, EE:s_ij})
            FF[:,j,i] = FF[:,i,j]
    return FF, pp

if __name__ == "__main__":        
    print('Training stats....',EXPERIMENT)
    
    sess=tf.InteractiveSession()
    sess.run(init_op)
    record = [];step = 0
    
    import time
    start = time.time()
    COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]
    j_record = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_cost_test = 0.
        for i in range(total_batch):
            train_xs,_ = train_data.next_batch(batch_size) 
            FF, pp = compt_F(sess, train_xs, pai, z_concate)                
            _,cost = sess.run([train_op,gen_loss],{x:train_xs,lr:np_lr, F:FF, PAI:pp})
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


