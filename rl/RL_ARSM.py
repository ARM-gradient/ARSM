import numpy as np
import gym
import sys
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tensorflow as tf
from arm_util import *
import matplotlib.pyplot as plt
import random


############################## one function ##################################
def pseudo_trajectory(pseudo_action, seed, pseudo_step, nstep,actions,rand_seed):
    # record the rewards till the pseudo step
    np.random.seed(rand_seed)
    pseudo_reward = rewards_true[:pseudo_step]        
    # use same seed as true trajectory
    pseudo_env.seed(seed)
    _ = pseudo_env.reset()
    # run until the pseudo step
    if pseudo_step != 0:
        history_action = actions[:pseudo_step]
        for a in history_action:
            _ = pseudo_env.step(a) # until the most recent state    
    next_state,reward,done,_ = pseudo_env.step(pseudo_action)    
    pseudo_reward.append(reward)
    cnt = 0 # count how many steps have run    
    while not done and pseudo_step+cnt+1 < nstep:
        state = next_state[None, :]
        phi = np.array(model_actor(state)[0])        
        if Share_Pi:
            #Share Pi between different pseudo actions, might reduce variance but also reduce exploration
            #Default is False
            if pseudo_step+cnt+1<len(pi_sequence): #n_true_:
                pi = pi_sequence[pseudo_step+cnt+1]
            else:
                pi = np.random.dirichlet(np.ones(nA))
        else:
            pi = np.random.dirichlet(np.ones(nA))
                
        pseudo_action = np.argmin(np.log(pi) - phi)
        next_state,reward,done,_ = pseudo_env.step(pseudo_action)
        pseudo_reward.append(reward)
        cnt += 1
        if done:
            break
        
    dr = discount_reward(pseudo_reward, gamma)
    return dr, np.sum(pseudo_reward), pseudo_step, cnt+1
###############################################################################



tf.enable_eager_execution()

#seedi = np.int(sys.argv[3])
seedi = 0
tf.set_random_seed(seedi)
np.random.seed(seedi)
random.seed(seedi)

#env_name = 'Acrobot-v1'
env_name = 'CartPole-v0'
#env_name = 'LunarLander-v2'


env = gym.make(env_name)
env_performance = gym.make(env_name) 
pseudo_env = gym.make(env_name)
nA = env.action_space.n
nS = env.observation_space.shape[0]

model_actor = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = "relu", input_shape = (nS,),\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)),
        tf.keras.layers.Dense(10, activation = "relu",\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)), 
        tf.keras.layers.Dense(nA, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
        ])
                                
num_epoch = 900

MaxPseudoActionSequences = 16

Num_ARSM_Ref_Episode = 0 
# if iter<Num_ARSM_Ref_Episode, then use ARSM single reference estimator 
# with the true action actegory or a random category as ref 
TrueActionAsRef = False

lr_actor = 0.03

entropy_par = 0.0
optimizer_actor = tf.train.AdamOptimizer(learning_rate=lr_actor)

gamma = .99 #reward discount factor

nstep = 3000 #maximum number of true + sudo actions
n_true = 3000 #maximum number of true actions

Share_Pi=False

IsPlot=False

SaveModel = True

score_record, entropy_record = [],[]
pseudo_prop = []

for e in range(num_epoch):
    ##Sample true action sequence
    seed = np.random.randint(0, 1e+9)
    env.seed(seed)
    state = env.reset()[None,:]
    rewards_true, actions, states, next_states = [],[],[],[]
    pseudo_action_sequences, phi_sequence, pi_sequence, Ref_cat_sequence=[],[],[],[]
    
    n_true_ = 0
    while (n_true_ < n_true):
        phi = model_actor(state)[0]
        phi_sequence.append(phi)
                                               
        pi = np.random.exponential(np.ones(nA))
        pi = pi/np.sum(pi)        
        pi_sequence.append(pi)
        
        action_true = np.argmin(np.log(pi) -phi)
        actions.append(action_true)
        states.append(state)
        next_state,reward,done,_ = env.step(action_true)
        next_states.append(next_state[None,:])
        state = next_state[None,:]
        
        
        rewards_true.append(reward)
        n_true_ += 1
        if done:
            break
    if Share_Pi:
        #Default is False
        for t in range(MaxPseudoActionSequences):
            pi = np.random.dirichlet(np.ones(nA))
            pi_sequence.append(pi)      
                              
    states = np.vstack(states)
    next_states = np.vstack(next_states)
    
    total_reward = discount_reward(rewards_true, gamma)
    
    
    
    ## Sample psuedo action sequences
    NumPseudoActionSequences=0
    unique_pseudo_actions = []
    if MaxPseudoActionSequences<1e6:
        time_permute=np.random.permutation(n_true_)
    else:
        time_permute=range(n_true_)
        
    time_permute_used=[]
    time_permute_unused=[]
    for t in time_permute:
        action_true=actions[t]
        pi = pi_sequence[t]
        phi = phi_sequence[t]
        if e<Num_ARSM_Ref_Episode:
            if TrueActionAsRef:
                Ref_cat=actions[t]
            else:
                Ref_cat=np.random.randint(nA)
                Ref_cat_sequence.append(Ref_cat)  
            pseudo_actions = pseudo_action_swap_vector(pi,phi.numpy(),Ref_cat)
        else:
            pseudo_actions = pseudo_action_swap_matrix(pi,phi.numpy())       
        temp = np.unique(pseudo_actions[pseudo_actions!=action_true])

        if NumPseudoActionSequences+temp.size>MaxPseudoActionSequences:
            break
        else:
            NumPseudoActionSequences =NumPseudoActionSequences+temp.size
        if temp.size>0:
            pseudo_action_sequences.append(pseudo_actions)
            temp = np.insert(temp[:,np.newaxis], 0, values=t, axis=1)
            unique_pseudo_actions.append(temp)
            time_permute_used.append(t)
        #else:
        #    time_permute_unused.append(t)
        
    pseudo_prop.append(len(unique_pseudo_actions)/n_true_)
    if len(unique_pseudo_actions)>0:    
        unique_pseudo_actions = np.vstack(unique_pseudo_actions)
        RandSeed = np.random.randint(0,1e9, NumPseudoActionSequences)
        ncpu = multiprocessing.cpu_count()
        pool = ProcessPoolExecutor(ncpu)
        futures = [pool.submit(pseudo_trajectory, pseudo_action,seed,pseudo_step,
                               nstep,actions,rand_seed) for pseudo_action,pseudo_step,rand_seed in \
                   zip(unique_pseudo_actions[:NumPseudoActionSequences,1],
                       unique_pseudo_actions[:NumPseudoActionSequences,0], RandSeed)]
        pseudo_sequences = [futures[i].result() for i in range(len(futures))]
        
        
        pseudo_sequences = np.vstack(pseudo_sequences)
        pseudo_reward_total_no_discount=pseudo_sequences[:,1]
        pseudo_reward_total = pseudo_sequences[:,0]
        if IsPlot:
            plt.subplot(321)
            plt.plot(pseudo_reward_total-total_reward)
                        
        f = np.zeros((n_true_,nA))
        t_cnt=0
        ft_collect=[]
        for t in time_permute_used:
            if e<Num_ARSM_Ref_Episode:
                ft = np.full(nA,total_reward)
            else:
                ft = np.full((nA,nA),total_reward)
            idxt=np.where(unique_pseudo_actions[:,0]==t)[0]            

            for idx in idxt:
                aa = unique_pseudo_actions[idx,1]
                ft[pseudo_action_sequences[t_cnt]==aa]=pseudo_reward_total[idx]
            
            meanft = np.mean(ft,axis=0)
        
            if e<Num_ARSM_Ref_Episode:
                if TrueActionAsRef:
                    Ref_cat=actions[t]
                else:
                    Ref_cat=Ref_cat_sequence[t_cnt]
                f[t,:]=(ft[Ref_cat]-np.mean(ft))*(1.0-nA*pi_sequence[t][Ref_cat])
            else:
                f[t,:]=np.matmul(ft-meanft,1.0/nA-pi_sequence[t])

            f[t,:]=f[t,:]/np.power(gamma,t)  #reward discount starting at time t+1
            
            t_cnt += 1

        if IsPlot:
            plt.subplot(323)
            plt.plot(f,'.')

        
        if e<25 and np.mean(np.abs(f[:n_true_,:]))<1e-14 :
            #Woth a bad initilization, ARSM generates no psudo actions and hence has zero gradient; so reinitilize the model
            model_actor = tf.keras.Sequential([
                    tf.keras.layers.Dense(10, activation = "relu", input_shape = (nS,),\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)),
                    tf.keras.layers.Dense(10, activation = "relu",\
                              kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0.1)), 
                    tf.keras.layers.Dense(nA, bias_initializer=tf.constant_initializer(0.1), kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3))
                    ])
        
        f_delta = tf.convert_to_tensor(-f, dtype=tf.float32)         
        grad_actor = gradient_arm(model_actor, states, f_delta,entropy_par)    
        optimizer_actor.apply_gradients(zip(grad_actor, model_actor.variables))
        
    ###### estimate variance ########
    vectorized_grads = tf.concat([tf.reshape(g, [-1]) for g in grad_actor if g is not None], axis=0)
    if e == 0:
        n_tmp = vectorized_grads.shape[0]
        first_moment_mean = np.zeros((n_tmp,))
        second_moment_mean = np.zeros((n_tmp,))
                
    vectorized_grads_sq = tf.square(vectorized_grads)
    first_moment_mean = (1 - 0.01) * first_moment_mean + 0.01 * vectorized_grads
    second_moment_mean = (1 - 0.01) * second_moment_mean + 0.01 * vectorized_grads_sq
    variance = second_moment_mean - tf.square(first_moment_mean)
    mean = np.array(tf.reduce_mean(tf.square(first_moment_mean)))
    variance = np.array(tf.reduce_mean(tf.log(variance+1e-20)))

    score = evaluate(model_actor, env_performance, nA, seed = np.random.randint(0,1e9))           
    score_record.append(score)

    ### calculate entropy
    logit_ent = model_actor(states)
    
    logprob_ent = logit_ent-tf.reduce_logsumexp(logit_ent,1,keepdims=True)
    prob_ent = tf.exp(logprob_ent)
    entropy_record.append(np.float(tf.reduce_sum(-prob_ent*logprob_ent))/n_true_)
    
    
    print("EP: " + str(e) + " Current_Score: " + str(score)+'\n')
    print("EP: " + str(e) + " Smooth_Score: " + str(np.mean(score_record[-100:])) + "         ",end="\n")

