
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import cPickle

#from OneHotCategorical import *
#from RelaxedOneHotCategorical import *

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical





#%%
directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
np_lr = 0.0001        
EXPERIMENT = 'Gumbel_l2'

batch_size = 200

training_epochs = 1000

tau0=1.0 # initial temperature

K=10
N=200//K

b_dim = 200
x_dim = 784

straight_through=True # if True, use Straight-through Gumbel-Softmax
kl_type='gumbel' # choose between ('relaxed', 'categorical')
learn_temp = True


def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder1(x,b_dim,reuse=False):
    with tf.variable_scope("encoder1", reuse = reuse):
        h1 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        log_alpha1 = tf.layers.dense(h1, b_dim, activation=None)
    return log_alpha1

def encoder2(y1,b_dim,reuse=False):
    with tf.variable_scope("encoder2", reuse = reuse):
        log_alpha2 = tf.layers.dense(y1, b_dim, activation=None)
    return log_alpha2


def decoder1(y1,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder1", reuse = reuse):
        h_x = slim.stack(y1 ,slim.fully_connected,[256,512],activation_fn=lrelu)
        log_alpha_x = tf.layers.dense(h_x, x_dim, activation=None)
    return log_alpha_x

def decoder2(y2,b_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder2", reuse = reuse):
        log_alpha1 = tf.layers.dense(y2, b_dim, activation=None)
    return log_alpha1

#%%
eps = 1e-10

tf.reset_default_graph() 

x0 = tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
x = tf.to_float(x0 > .5)

logits_y = tf.reshape(encoder1(x,b_dim),[-1,N,K])

tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()
y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
y = slim.flatten(tf.stop_gradient(y_hard - y) + y)


logits_y2 = tf.reshape(encoder2(y,b_dim),[-1,N,K])
q_y2 = RelaxedOneHotCategorical(tau,logits_y2)
y2 = q_y2.sample()
y_hard2 = tf.cast(tf.one_hot(tf.argmax(y2,-1),K), y.dtype)
y2 = slim.flatten(tf.stop_gradient(y_hard2 - y2) + y2)



logits_y1 = tf.reshape(decoder2(y2,b_dim),[-1,N,K])

logits_x = decoder1(y,x_dim)

p_x = Bernoulli(logits=logits_x)
x_mean = p_x.mean()


recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K #uniform



def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-5
    q = tf.nn.softmax(q_logit,dim=2)
    p = tf.nn.softmax(p_logit,dim=2)
    return tf.reduce_sum(q*(tf.log(q+eps)-tf.log(p+eps)),axis = [1,2])

def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))



probq_z1 = tf.nn.softmax(logits_y, -1)
logq_z1x = tf.reduce_sum(tf.log(y_hard * probq_z1 + eps),axis = [1,2])


KL = kl_cat(logits_y2, logits_py) 

log_p_x_given_z1 = bernoulli_loglikelihood(x, logits_x)
log_p_x_given_z1 = tf.reduce_sum(log_p_x_given_z1, axis=1)

probp_z1 = tf.nn.softmax(logits_y1, -1)
log_p_z1_given_z2 = tf.reduce_sum(tf.log(y_hard * probp_z1 + eps),axis = [1,2])

neg_elbo0 = logq_z1x - log_p_z1_given_z2 - log_p_x_given_z1 + KL



lr=tf.constant(0.0001)

neg_elbo = neg_elbo0[:,np.newaxis]

loss = tf.reduce_mean(neg_elbo)



train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
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
        xs, _ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(neg_elbo0,{x:xs}))
    return np.mean(cost_eval)


print('Training starts....',EXPERIMENT)

sess=tf.InteractiveSession()
sess.run(init_op)
record = [];step = 0



    
import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]
evidence_r = []

for epoch in range(training_epochs):
    
    record = [];
    
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)   
        _,cost,_ = sess.run([train_op,loss,tau],{x0:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
    
    print(epoch,'cost=',np.mean(record),'with std=',np.std(record))
    
    if epoch%1 == 0:
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(get_loss(sess,valid_data,total_valid_batch))
    if epoch%5 == 0:
#        avg_evi_val = evidence(sess, valid_data, -neg_elbo, batch_size, S = 100, total_batch=10)
#        print(epoch,'The validation NLL is', -np.round(avg_evi_val,2))
#        evidence_r.append(np.round(avg_evi_val,2))        
#
#        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list,evidence_r]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'wb'))
            
#avg_evi_test = evidence(sess, test_data, -neg_elbo, batch_size, S = 1000)
#print("The final test NLL is", -np.round(avg_evi_test,2))
#cPickle.dump([np_lr] + all_ + [avg_evi_test], open(directory+EXPERIMENT, 'wb'))



print(EXPERIMENT)












