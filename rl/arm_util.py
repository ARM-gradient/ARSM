
import numpy as np
import tensorflow as tf
import copy
import scipy.stats

import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def pseudo_action_swap_matrix(pi,phi):
    C=len(pi)
    RaceAllSwap = np.log(pi[:,np.newaxis])-phi[np.newaxis,:]
    Race = np.diag(RaceAllSwap)
    action_true = np.argmin(Race)

    #tic()
    if C<=6: # True: #True:
        #Slow version for large C
        pseudo_actions=np.full((C, C), action_true)
        for m in range(C):
            for jj in  range(m):
                RaceSwap = Race.copy()
                RaceSwap[m], RaceSwap[jj]=RaceAllSwap[jj,m],RaceAllSwap[m,jj]
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m,jj], pseudo_actions[jj,m] = s_action, s_action

    else:
        Race_min = Race[action_true]
        #Fast version for large C
        pseudo_actions=np.full((C, C), action_true)
        SwapSuccess = RaceAllSwap<=Race_min
        SwapSuccess[action_true,:]=True
        np.fill_diagonal(SwapSuccess,0)
        m_idx,j_idx = np.where(SwapSuccess)
        for i in range(len(m_idx)):
            m,jj = m_idx[i],j_idx[i]
            RaceSwap = Race.copy()
            RaceSwap[m], RaceSwap[jj]=RaceAllSwap[jj,m],RaceAllSwap[m,jj]
            if m==action_true or jj == action_true:
                s_action = np.argmin(RaceSwap)
                pseudo_actions[m,jj], pseudo_actions[jj,m] = s_action, s_action
            else:
                if RaceSwap[m]<RaceSwap[jj]:
                    pseudo_actions[m,jj], pseudo_actions[jj,m] = m, m
                else:
                    pseudo_actions[m,jj], pseudo_actions[jj,m] = jj, jj
    return pseudo_actions

def pseudo_action_swap_vector(pi,phi,Cat_ref):
    C=len(pi)
    
    Race = np.log(pi)-phi
    action_true=np.argmin(Race)
    pseudo_actions=np.full(C,action_true)
    for m in range(C):
        jj=Cat_ref
        RaceSwap = Race.copy()
        if m!=jj:
            RaceSwap[m] = np.log(pi[jj])-phi[m]
            RaceSwap[jj] = np.log(pi[m])-phi[jj]
        pseudo_actions[m] = np.argmin(RaceSwap)
    return pseudo_actions


def loss_reinforce(model, states, labels, drs, ent = False): 
    logit = model(states)
    if ent:
        probability = tf.nn.softmax(logit)
        entropy = scipy.stats.entropy(probability)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=labels) * drs - entropy * 0.01)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=labels) * drs)
    
def loss_reinforce_batch(model, states, actions, advantages):
    logit = model(states)
    prob = tf.nn.softmax(logit)
    ent = tf.nn.softmax_cross_entropy_with_logits(labels = prob, logits = logit)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=actions) * tf.squeeze(advantages)) - \
            tf.reduce_mean(0.01 * ent)

def gradient_reinforce_batch(model, states, actions, advantages):
    with tf.GradientTape() as tape:
        loss_fn = loss_reinforce_batch(model, states, actions, advantages)
    return tape.gradient(loss_fn, model.variables)    

def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

def loss_arm(model, states, grad_alpha, ent_par):
    # grad_alpha: T * num_of_actions
    logit = model(states)
    
    logprob = logit-tf.reduce_logsumexp(logit,1,keepdims=True)
    prob = tf.exp(logprob)
    ent = tf.reduce_sum(-prob*logprob)
    
    if ent_par>0:        
        return tf.reduce_sum(tf.multiply(logit, grad_alpha)) - ent_par*ent
    else:
        return tf.reduce_sum(tf.multiply(logit, grad_alpha))
    

def loss_critic(model, state, drs):
    return tf.reduce_mean(tf.square(drs - model(state)))

def gradient_reinforce(model, states, actions, drs, ent = False):
    with tf.GradientTape() as tape:
        loss_fn = loss_reinforce(model, states, actions, drs, ent)
    return tape.gradient(loss_fn, model.variables)

def gradient_arm(model, states, grad_alpha, ent_par):
    with tf.GradientTape() as tape:
        loss_fn = loss_arm(model, states, grad_alpha, ent_par)
    return tape.gradient(loss_fn, model.variables)




def gradient_critic(model, state, drs):
    with tf.GradientTape() as tape:
        loss_fn = loss_critic(model, state, drs)
    return tape.gradient(loss_fn, model.variables)

def discount_reward(rewards, gamma): # no normalization
    dr = np.sum(np.power(gamma,np.arange(len(rewards)))*rewards)
    return dr

def discount_rewards(rewards, gamma): # no normalization
    drs = np.sum(np.power(gamma,np.arange(len(rewards)))*rewards)[None]
    return drs

def swap(array, a,b):
    array[a], array[b] = array[b], array[a]
    return array

def evaluate(model_actor, env, nA, seed):
    env.seed(seed)
    state = env.reset()[None,:]
    state = np.float32(state)
    score = 0    
    while True:
        logits = np.array(model_actor(state)[0])
        pi = np.random.dirichlet(np.ones(nA))
        #action = np.argmin(pi * np.exp(-logits))
        action = np.argmin(np.log(pi)-logits)
    
        next_state,reward,done,_ = env.step(action)
        next_state = np.float32(next_state)
        next_state = next_state[None,:] ## add one layer on the structure, e.g.[1,2,3] to [[1,2,3]]
    
        # Compute gradient and save with reward in memory for our weight update
        score += reward
    
        # Dont forget to update your old state to the new state
        state = next_state
    
        if done:
            break 
    return score

def Q_value(model_critic, state, action, nA):
    action_one_hot = tf.one_hot(action, nA, 1.0, 0.0)
    q = model_critic(state)
    pred = tf.reduce_sum(q * action_one_hot, reduction_indices=-1)
    return pred

def loss_critic_q(model_critic, states, actions, drs, nA,Prob,rewards,gamma,model_actor,unique_pseudo_actions,pseudo_action_sequences,pi_sequence,time_permute_used,n_true_,e):
    action_one_hot = tf.one_hot(actions, nA, 1.0, 0.0)
    q = model_critic(tf.concat([states,action_one_hot], 1))
    q_values = q
    phi = model_actor(states)
    
    Prob1 = tf.nn.softmax(phi)

    q_values_next=0
    for aa in range(nA):
        action_one_hot = tf.one_hot(tf.fill((len(actions),),aa), nA, 1.0, 0.0)
        q_values_next+=(model_critic(tf.concat([states,action_one_hot], 1)))*Prob1[:,aa][:,None]

    q_values_next = tf.stop_gradient(q_values_next)

    
    pseudo_action_one_hot = tf.one_hot(unique_pseudo_actions[:,1], nA, 1.0, 0.0)

    pseudo_reward_total = model_critic(tf.concat([states[unique_pseudo_actions[:,0]],pseudo_action_one_hot], 1))
    f = []
    ttt=0
    #for t in time_permute_used:
    for t in range(n_true_):
        if len(np.where(t==time_permute_used)[0])>0:
            if t<n_true_-1:
                total_reward = rewards[t]+ gamma*tf.reduce_sum(model_critic(states[t+1][None])*Prob[t+1])
            else:
                total_reward = rewards[t]
            ft = tf.ones((nA,nA)) * total_reward
            idxt=np.where(unique_pseudo_actions[:,0]==t)[0]            
            for idx in idxt:
                aa = unique_pseudo_actions[idx,1]
                matrix_tmp = tf.to_float(pseudo_action_sequences[ttt]==aa) * (pseudo_reward_total[idx] - total_reward)
                ft = ft + matrix_tmp
            
            meanft = tf.reduce_mean(ft,axis=0)
            sec_tmp = tf.convert_to_tensor(1.0/nA-pi_sequence[t], dtype = tf.float32)
            sec_tmp = tf.reshape(sec_tmp, (1,-1))
            f.append(tf.matmul(sec_tmp, ft-meanft)) # make it a row vector
            ttt+=1
        else:
            f.append(tf.zeros(nA)) 
    
    f1 = tf.stack(f, axis=0)
    f = tf.reshape(f1, (-1,nA))
    logit = model_actor(states)    
    var_grad = tf.reduce_sum(tf.square(tf.multiply(logit, f)))

    return tf.reduce_sum(tf.square(q_values[:-1]-tf.convert_to_tensor(rewards[:-1],dtype = tf.float32)[:,None]-gamma*q_values_next[1:]))\
  + tf.reduce_sum(tf.square(q_values[:-2]-tf.convert_to_tensor(rewards[:-2],dtype = tf.float32)[:,None]-gamma*tf.convert_to_tensor(rewards[1:-1],dtype = tf.float32)[:,None]-gamma**2*q_values_next[2:]))*0.8\
   + tf.reduce_sum(tf.square(drs[:,None] - q_values))*0.3
    
def gradient_critic_q(model_critic, states, actions, drs, nA,Prob,rewards,gamma,model_actor,unique_pseudo_actions,pseudo_action_sequences,pi_sequence,time_permute_used,n_true_,e):
    with tf.GradientTape() as tape:
        loss_fn = loss_critic_q(model_critic, states, actions, drs, nA,Prob,rewards,gamma,model_actor,unique_pseudo_actions,pseudo_action_sequences,pi_sequence,time_permute_used,n_true_,e)
    return tape.gradient(loss_fn, model_critic.variables)


def gradient_actor_q(model_critic, states, actions, drs, nA,Prob,rewards,gamma,model_actor):
    with tf.GradientTape() as tape:
        loss_fn = loss_critic_q(model_critic, states, actions, drs, nA,Prob,rewards,gamma,model_actor)
    return tape.gradient(loss_fn, model_actor.variables)


def loss_critic_sa(model_critic_sa, states, actions, drs, nA):
    action_one_hot = tf.one_hot(actions, nA, 1.0, 0.0)

    q= model_critic_sa(tf.concat([states,action_one_hot], 1))
    
    q_values = tf.reduce_sum(q * action_one_hot, reduction_indices=1)
    
    q_values_next = q
    
    
    
    return tf.reduce_mean(tf.square(drs[:,None] - q_values))  

def gradient_critic_sa(model_critic_sa, states, actions, drs, nA):
    with tf.GradientTape() as tape:
        loss_fn = loss_critic_sa(model_critic_sa, states, actions, drs, nA)
    return tape.gradient(loss_fn, model_critic_sa.variables)


def policy(model_actor, state, nA):
    logits = np.array(model_actor(state)[0])
    pi = np.array([np.random.dirichlet(np.ones(nA)) for i in range(len(logits))])
    action = np.argmin(pi * np.exp(-logits), axis = 1) # 1 is row-wise and 0 is column-wise    
    return action

def loss_dqn(model_critic,y, states, actions, nA, Q_value):
    x = Q_value(model_critic, states, actions, nA)
    return tf.reduce_sum(tf.square(x-y))

def gradient_dqn(model_critic, y, states, actions, nA,Q_value):
    with tf.GradientTape() as tape:
        loss_fn = loss_dqn(model_critic,y,states,actions,nA, Q_value)
    return tape.gradient(loss_fn, model_critic.variables)



