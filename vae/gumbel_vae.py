#ST-Gumbel-Softmax
import tensorflow as tf
import numpy as np
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
EXPERIMENT = 'gumbel_softmax'

batch_size = 200
training_epochs = 1000
lr = 0.0001
x_dim = 784
z_dim = 200
K = 10     #number of cat variable
C = z_dim//K  #C-way
learn_temp = True

def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def encoder(x,z_dim):
    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
        h = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h, z_dim, activation=None)
    return log_alpha

def decoder(b,x_dim):
    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
        h = slim.stack(b ,slim.fully_connected,[256,512],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h, x_dim, activation=None)
    return log_alpha

#%%
tf.reset_default_graph() 

x0 = tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
x = tf.to_float(x0 > .5)

logits_y = tf.reshape(encoder(x,z_dim),[-1,C,K])
tau = tf.Variable(1.0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()

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

KL = tf.reduce_sum(KL_qp,1)

neg_elbo0 = KL - recons
neg_elbo = neg_elbo0[:,np.newaxis]
loss = tf.reduce_mean(KL - recons)

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
        cost_eval.append(sess.run(neg_elbo0,{x0:xs}))
    return np.mean(cost_eval)

print('Training starts....',EXPERIMENT)

sess=tf.InteractiveSession()
sess.run(init_op)
step = 0
  
COUNT=[]; epoch_list=[]; COST=[]; COST_TEST=[]; COST_VALID=[]; record = [];

for epoch in range(training_epochs):      
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)   
        _,cost,_ = sess.run([train_op,loss,tau],{x0:train_xs})
        record.append(cost)
        step += 1
        
    if epoch%1 == 0:
        COUNT.append(step); COST.append(np.mean(record));
        val_loss = get_loss(sess,valid_data,total_valid_batch)
        COST_VALID.append(val_loss)
        print(epoch,'train_loss=',np.mean(record),'val_loss=', val_loss)
    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        all_ = [COUNT,COST,COST_TEST,COST_VALID,epoch_list]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'wb'))
    record = []
            

