
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf

tf.enable_eager_execution()

i=5
np.random.seed(i)
random.seed(i)


def softmax(phi):
    e_phi = np.exp(phi - np.max(phi))
    return e_phi / e_phi.sum(axis=0) 

def fun(z,C,r):
    return .5+(z+1)/C/r


def loss_relax(phi, u1, u2, model_phi):
    theta = tf.nn.softmax(phi)
    z = tf.log(theta) - tf.log(-tf.log(u1))
    b = tf.argmax(z)
    b_onehot = tf.one_hot(b,C)
    b_onehot = b_onehot * -1
    b_onehot = b_onehot + 1 # make everywhere 1 except for i = b
    tmp = -b_onehot * tf.log(u2) / theta
    z_tilde = -tf.log(tmp - tf.log(u2))
    logp = phi[b] - tf.log(tf.reduce_sum(tf.exp(phi)))
    loss = model_phi(z[None,:]) - model_phi(z_tilde[None,:]) + \
    (tf.convert_to_tensor(Fun[b],dtype=tf.float32) - model_phi(tf.stop_gradient(z_tilde[None,:]))) * logp
    return loss


def grad_relax(phi, u1, u2, model_phi):
    with tf.GradientTape() as tape:
        loss_fn = loss_relax(phi, u1, u2, model_phi)
    return tape.gradient(loss_fn, phi)

def grad_phi(phi,u1,u2,model_phi):
    with tf.GradientTape() as tape:
        pg_grads = grad_relax(phi,u1,u2, model_phi)
#        cv_grads = tf.concat([tf.reshape(p, [-1]) for p in pg_grads], 0)
        loss_fn = tf.reduce_sum(tf.square(pg_grads))
    return tape.gradient(loss_fn, model_phi.variables)    





C=30
r=30

model_phi = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = "relu", input_shape = (C,),\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)),
        tf.keras.layers.Dense(10, activation = "relu",\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)),    
        tf.keras.layers.Dense(1, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
        ])
optimizer = tf.train.AdamOptimizer(0.001)

phi_RELAX = np.zeros(C)
phi_RELAX_record = []
prob_RELAX_record = []
reward_expected_RELAX_record = []
grad_RELAX_record = []
VAR_RELAX = []
snr_RELAX_record = []

IterMax=5000

stepsize = 1

f = np.zeros(C)

Fun = fun(np.arange(C),C,r)

for iter in range(IterMax):
    phi_RELAX = tf.contrib.eager.Variable(phi_RELAX, dtype = tf.float32)
    u1 = tf.random_uniform(shape = (C,))
    u2 = tf.random_uniform(shape = (C,))
    grad_RELAX = grad_relax(phi_RELAX, u1, u2, model_phi)
    
    optimizer.apply_gradients(zip(grad_phi(phi_RELAX,u1,u2,model_phi), model_phi.variables))    
    
    phi_RELAX = phi_RELAX + stepsize/2 * grad_RELAX # want to maximize, so plus
    
    prob_RELAX = softmax(phi_RELAX)
    reward_expected_RELAX = np.sum(prob_RELAX*Fun)    
    
    phi_RELAX_record.append(phi_RELAX)
    prob_RELAX_record.append(prob_RELAX)
    reward_expected_RELAX_record.append(reward_expected_RELAX)
    grad_RELAX_record.append(np.array(grad_RELAX))
    if iter%100 == 0:
        print("Iter: " + str(iter))
    if iter % 100 == 0:
        var_relax = []
        for j in range(100):
            phi_RELAX = tf.contrib.eager.Variable(phi_RELAX, dtype = tf.float32)
            u1 = tf.random_uniform(shape = (C,))
            u2 = tf.random_uniform(shape = (C,))
            grad_RELAXz = grad_relax(phi_RELAX, u1, u2, model_phi)            
            var_relax.append(np.array(grad_RELAXz))
            
        VAR_RELAX.append(np.mean(np.var(var_relax,axis=0))) 
        mean_relax = np.mean(var_relax,axis=0)
        std_relax = np.std(var_relax,axis=0)
        snr_RELAX_record.append(np.abs(mean_relax)/std_relax)
    
##### relax
plt.subplot(2,3,1)  
plt.imshow(prob_RELAX_record,aspect='auto')
plt.colorbar()
plt.title("RELAX prob")

plt.subplot(2,3,2)
plt.semilogy(prob_RELAX_record);
plt.title("RELAX prob")

plt.subplot(2,3,3)  
plt.imshow(grad_RELAX_record,aspect='auto')
plt.colorbar()
plt.title("RELAX gradient")

plt.subplot(2,3,4)  
plt.plot(grad_RELAX_record);
plt.title("RELAX gradient")
plt.tight_layout()
plt.show()

plt.subplot(2,3,5)  
plt.plot(VAR_RELAX);
plt.title("RELAX gradient variance")
plt.tight_layout()
plt.show()

plt.subplot(2,3,6)  
plt.plot(reward_expected_RELAX_record);
plt.title("RELAX expected reward")
plt.tight_layout()
plt.show()
#####

#write
import pickle
with open('/relax_results/prob_RELAX_record'+str(C), 'wb') as f:
    pickle.dump(prob_RELAX_record, f, protocol=2)
        
with open('/relax_results/grad_RELAX_record'+str(C), 'wb') as f:
    pickle.dump(grad_RELAX_record, f, protocol=2) 

with open('/relax_results/VAR_RELAX'+str(C), 'wb') as f:
    pickle.dump(VAR_RELAX, f, protocol=2)         

with open('/relax_results/reward_expected_RELAX_record'+str(C), 'wb') as f:
    pickle.dump(reward_expected_RELAX_record, f, protocol=2)  

with open('/relax_results/snr_RELAX_record'+str(C), 'wb') as f:
    pickle.dump(snr_RELAX_record, f, protocol=2)        