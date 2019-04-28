
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import cPickle


slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical


#%%
directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
np_lr = 0.0001        
EXPERIMENT = 'Gumbel-Softmax'

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



def encoder(x,b_dim,reuse=False):
    with tf.variable_scope("encoder", reuse = reuse):
        h2 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, b_dim, activation=None)
    return log_alpha


def decoder(b,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder", reuse = reuse):
        h2 = slim.stack(b ,slim.fully_connected,[256,512],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h2, x_dim, activation=None)
    return log_alpha

#%%


tf.reset_default_graph() 

x0 = tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
x = tf.to_float(x0 > .5)

logits_y = tf.reshape(encoder(x,b_dim),[-1,N,K])

tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()
#if straight_through:
y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
y = tf.stop_gradient(y_hard - y) + y
net = slim.flatten(y)

logits_x = decoder(net,x_dim)

p_x = Bernoulli(logits=logits_x)
x_mean = p_x.mean()


recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K #uniform

p_cat_y = OneHotCategorical(logits=logits_py)
q_cat_y = OneHotCategorical(logits=logits_y)
KL_qp =  tf.distributions.kl_divergence(q_cat_y, p_cat_y)

lr=tf.constant(0.0001)

KL = tf.reduce_sum(KL_qp,1)

mean_recons = tf.reduce_mean(recons)
mean_KL = tf.reduce_mean(KL)

neg_elbo0 = KL - recons

neg_elbo = neg_elbo0[:,np.newaxis]

loss = -tf.reduce_mean(recons - KL)


gs_grad = tf.gradients(loss, logits_y)





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

if __name__ == "__main__": 
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
            _,cost,_ = sess.run([train_op,loss,tau],{x:train_xs,lr:np_lr})
            record.append(cost)
            step += 1
        
        print(epoch,'cost=',np.mean(record),'with std=',np.std(record))
        
        if epoch%1 == 0:
            COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
            COST_VALID.append(get_loss(sess,valid_data,total_valid_batch))
        if epoch%5 == 0:
            epoch_list.append(epoch)
            time_list.append(time.time()-start)
            all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list,evidence_r]
            cPickle.dump(all_, open(directory+EXPERIMENT, 'wb'))
 
    print(EXPERIMENT)












