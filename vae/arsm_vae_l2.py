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
        

def encoder1(x,z_dim,reuse=False):
    #return logits [N,n_cv*(n_class-1)]
    #z_dim is n_cv*(n_class)
    with tf.variable_scope("encoder1") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu,scope='e1')
        logit_z1 = tf.layers.dense(h2, z_dim, name="encoder1_out",activation = None)
    return logit_z1

def encoder2(z1,z_dim,reuse=False):
    #return logits [N,n_cv*(n_class-1)]
    #z_dim is n_cv*(n_class)
    with tf.variable_scope("encoder2") as scope:
        if reuse:
            scope.reuse_variables()
        z1 = slim.flatten(z1)  
        logit_z2 = tf.layers.dense(z1, z_dim, name="encoder2_out",activation = None)
    return logit_z2

def decoder1(b1,x_dim,reuse=False):
    #return logits
    #b is [N,K,n_cv,n_class]
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        b1 = slim.flatten(b1)   
        h2 = slim.stack(b1, slim.fully_connected,[256, 512],activation_fn=lrelu,scope='d1')
        logit_x = tf.layers.dense(h2, x_dim, name="decoder2_out",activation = None)
    return logit_x

def decoder2(b2,z_dim,reuse=False):
    #return logits
    #b is [N,K,n_cv,n_class]
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()        
        b2 = slim.flatten(b2)      
        logit_b1 = tf.layers.dense(b2, z_dim,name="decoder1_out", activation = None)
    return logit_b1
    

def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-5
    q = tf.nn.softmax(q_logit,dim=2)
    p = tf.nn.softmax(p_logit,dim=2)
    return tf.reduce_sum(q*(tf.log(q+eps)-tf.log(p+eps)),axis = [1,2])


def fun(x_star,E1,E2,reuse_decoder=False, reuse_encoder=False):
    '''
    x_star is N*d_x, E is N*n_cv*n_class
    x_star is observe x; E is latent b
    return (N,)
    '''
    #n_class = 10 ; n_cv = 20
    eps = 1e-6
    logitsq_z1 = encoder1(x_star,z_dim,reuse=reuse_encoder)
    logitsq_z1 = tf.reshape(logitsq_z1, [-1, n_cv, n_class])
    probq_z1 = tf.nn.softmax(logitsq_z1, -1)
    logq_z1x = tf.reduce_sum(tf.log(E1 * probq_z1 + eps),axis = [1,2])
    
    logitsq_z2 = encoder2(E1,z_dim,reuse=reuse_encoder)
    logitsq_z2 = tf.reshape(logitsq_z2, [-1, n_cv, n_class])
    logits_py = tf.ones_like(logitsq_z2)
    KL = kl_cat(logitsq_z2, logits_py) 
    
    #log p(x_star|E1)
    logit_x = decoder1(E1,x_dim,reuse=reuse_decoder)
    log_p_x_given_z1 = bernoulli_loglikelihood(x_star, logit_x)
    # (N,)
    log_p_x_given_z1 = tf.reduce_sum(log_p_x_given_z1, axis=1)
    
    logitsp_z1 = decoder2(E2,z_dim,reuse=reuse_decoder)
    logitsp_z1 = tf.reshape(logitsp_z1, [-1, n_cv, n_class])
    probp_z1 = tf.nn.softmax(logitsp_z1, -1)
    log_p_z1_given_z2 = tf.reduce_sum(tf.log(E1 * probp_z1 + eps),axis = [1,2])
    
    neg_elbo = logq_z1x - log_p_z1_given_z2 - log_p_x_given_z1 + KL
    return neg_elbo


#%%
EXPERIMENT = 'ARSM_l2'
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
z_dim = n_cv * n_class   # # of latent parameters neede for one cat var is n_cat-1

eps = 1e-10
K_u = 1; K_b = 1
lr=tf.constant(0.0001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#encoder q(b|x) = log Ber(b|log_alpha_b)
logit_z10 = encoder1(x_binary,z_dim) 
logit_z1 = tf.reshape(logit_z10, [-1, n_cv, n_class])
q_z1 = Categorical(logits=logit_z1) 
z1_sample = q_z1.sample() #N*n_cv
z1_sample = tf.one_hot(z1_sample,depth=n_class)
z1_sample = tf.cast(z1_sample,tf.float32) 

logit_z20 = encoder2(z1_sample ,z_dim) 
logit_z2 = tf.reshape(logit_z20, [-1, n_cv, n_class])
q_z2 = Categorical(logits=logit_z2) 
z2_sample = q_z2.sample() #N*n_cv
z2_sample = tf.one_hot(z2_sample,depth=n_class)
z2_sample = tf.cast(z2_sample,tf.float32) 

gen_loss0 = fun(x_binary,z1_sample,z2_sample,reuse_decoder= False, reuse_encoder=True)
#average over N
gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)




#provide encoder q(b|x) gradient by data augmentation
Dir_two = Dirichlet([1.0]*n_class)
pai_two = Dir_two.sample(sample_shape=[N,n_cv]) #[N,n_cv,n_class]

EE_two = tf.placeholder(tf.float32,[None, n_cv, n_class]) 
EE_two1 = tf.placeholder(tf.float32,[None, n_cv, n_class]) #z1_sample
F_two_ij = fun(x_binary,EE_two1,EE_two,reuse_decoder= True,reuse_encoder= True)
  
#compt_F2    
F_two = tf.placeholder(tf.float32,[None,n_class,n_class]) #n_class*n_class
F_two0 = F_two - tf.reduce_mean(F_two, axis = 2, keep_dims=True)
F_two1 = tf.expand_dims(F_two0, axis=1)
PAI_two = tf.placeholder(tf.float32,[None,n_cv,n_class])
pai2 = 1/n_class - tf.tile(tf.expand_dims(PAI_two, axis=2),[1,1,n_class,1])

alpha_grads2 = tf.reduce_mean(F_two1*pai2, axis = -1)
alpha_grads_two = tf.reshape(alpha_grads2,[-1,z_dim])


#gradient to log_alpha_b1(phi_1)       

Dir_one = Dirichlet([1.0]*n_class)
pai_one = Dir_one.sample(sample_shape=[N,n_cv]) #[N,n_cv,n_class]

EE_one = tf.placeholder(tf.float32,[None, n_cv, n_class]) 
logits_one2 = encoder2(EE_one,z_dim,reuse=True)
logits_one2 = tf.reshape(logits_one2, [-1, n_cv, n_class])

EE_one2 = (Categorical(logits=logits_one2)).sample()
EE_one2 = tf.one_hot(EE_one2,depth=n_class)
EE_one2 = tf.cast(EE_one2,tf.float32) 

#compt_F1
F_one_ij = fun(x_binary,EE_one,EE_one2,reuse_decoder= True,reuse_encoder= True)
F_one = tf.placeholder(tf.float32,[None,n_class,n_class]) #n_class*n_class
F_one0 = F_one - tf.reduce_mean(F_one, axis = 2, keep_dims=True)
F_one1 = tf.expand_dims(F_one0, axis=1)
PAI_one = tf.placeholder(tf.float32,[None,n_cv,n_class])
pai1 = 1/n_class - tf.tile(tf.expand_dims(PAI_one, axis=2),[1,1,n_class,1])

alpha_grads1 = tf.reduce_mean(F_one1*pai1, axis = -1)
alpha_grads_one = tf.reshape(alpha_grads1,[-1,z_dim])




inf_vars1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder1')
inf_grads1 = tf.gradients(logit_z10, inf_vars1, grad_ys=alpha_grads_one)
inf_gradvars1 = zip(inf_grads1, inf_vars1)
inf_opt1 = tf.train.AdamOptimizer(lr)
inf_train_op1 = inf_opt1.apply_gradients(inf_gradvars1)

inf_vars2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder2')
inf_grads2 = tf.gradients(logit_z20, inf_vars2, grad_ys=alpha_grads_two)
inf_gradvars2 = zip(inf_grads2, inf_vars2)
inf_opt2 = tf.train.AdamOptimizer(lr)
inf_train_op2 = inf_opt2.apply_gradients(inf_gradvars1)



with tf.control_dependencies([gen_train_op, inf_train_op1, inf_train_op2]):
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

def compt_F2(sess, train_xs, pai_two, logit_z2, z1_sample):
    pp, ph, z1 = sess.run([pai_two, logit_z2, z1_sample],{x:train_xs})
    FF = np.zeros([batch_size, n_class,n_class])
    from keras.utils.np_utils import to_categorical   
    for i in range(n_class):
        for j in range(i,n_class):
            pp_ij = np.copy(pp)
            pp_ij[:,:,[i,j]] = pp_ij[:,:,[j,i]]
            s_ij  = to_categorical(np.argmin(np.log(pp_ij+1e-6)-ph, axis = 2),num_classes=n_class)
            FF[:,i,j] = sess.run(F_two_ij,{x:train_xs, EE_two1:z1, EE_two:s_ij})
            FF[:,j,i] = FF[:,i,j]
    return FF, pp
        
def compt_F1(sess, train_xs, pai_one, logit_z1):
    pp, ph = sess.run([pai_one, logit_z1],{x:train_xs})
    FF = np.zeros([batch_size, n_class,n_class])
    from keras.utils.np_utils import to_categorical   
    for i in range(n_class):
        for j in range(i,n_class):
            pp_ij = np.copy(pp)
            pp_ij[:,:,[i,j]] = pp_ij[:,:,[j,i]]
            s_ij  = to_categorical(np.argmin(np.log(pp_ij+1e-6)-ph, axis = 2),num_classes=n_class)
            FF[:,i,j] = sess.run(F_one_ij,{x:train_xs, EE_one:s_ij})
            FF[:,j,i] = FF[:,i,j]
    return FF, pp

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
        FF2, pp2 = compt_F2(sess, train_xs, pai_two, logit_z2, z1_sample)   
        FF1, pp1 = compt_F1(sess, train_xs, pai_one, logit_z1) 
        plh_dict = {x:train_xs,lr:np_lr, F_two:FF2, PAI_two:pp2, F_one:FF1, PAI_one:pp1}                           
        _,cost = sess.run([train_op,gen_loss],plh_dict)
        record.append(cost)
        step += 1
        #print(cost)
        
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch)
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(valid_loss)
        print(epoch,'valid_cost=',valid_loss,'with std=',np.std(record))
        print(time.time()-start)
    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'w'))
    record=[]



print(EXPERIMENT)


