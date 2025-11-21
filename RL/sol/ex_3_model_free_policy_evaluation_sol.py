#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np

def mc_policy_eval(env, gamma, pi, nEpisodes, maxEpisodeLength, 
                   V_real, plot=False, nprint=1000):
    ''' Monte-Carlo Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # create a vector N to store the number of times each state has been visited
    N = np.zeros(env.nx)
    # create a vector C to store the cumulative cost associated to each state
    C = np.zeros(env.nx)
    # create a vector V to store the Value
    V = np.zeros(env.nx)
    # create a list V_err to store history of the error between real and estimated V table
    V_err = []
    # for each episode
    for k in range(nEpisodes):
        # reset the environment to a random state
        x = env.reset()
        # keep track of the states visited in this episode
        x_list = []
        # keep track of the costs received at each state in this episode
        cost_list = []
        # simulate the system using the policy pi   
        for t in range(maxEpisodeLength):
            if callable(pi):        # pi is a function
                u = pi(env, x)
            else:                # pi is a vector
                u = pi[x]
            x, cost = env.step(u)
            x_list.append(x)
            cost_list.append(cost)
        
        # Update the V-Table by computing the cost-to-go J backward in time   
        # J[t+1] = l(t+1) + gamma * l[t+2] + gamma^2 * l[t+3] + ...
        # J[t]   = l(t)   + gamma * J[t+1] + gamma^2 * J[t+2] + ...
        #        = l(t) + gamma * J[t+1]
        # Note: we do not need to recompute each time all the terms, we can just reuse the previous J
        J = 0        # discounted cost-to-go
        for t in range(maxEpisodeLength, -1, -1):   # from maxEpisodeLength to 0 
            J = cost_list[t] + gamma*J # at fist iteration, the cost is just cost_list[t], not discounted
            x = x_list[t]
            N[x] += 1   # increment the visit count for state x
            C[x] += J   # add the cost-to-go to the total cost for state x
            V[x] = C[x] / N[x]  # update the Value for state x
    
        # compute V_err as: mean(abs(V-V_real))
        V_err.append(np.mean(np.abs(V - V_real)))

        if(k % nprint == 0):
            print("MC iter %d, V_err = %.6f" % (k, V_err[-1]))
            print("Number of states not visited: %d", env.nx - np.count_nonzero(N))
            if(plot):
                env.plot_V_table(V)


    return V, V_err


def td0_policy_eval(env, gamma, pi, V0, nEpisodes, maxEpisodeLength, 
                    V_real, learningRate, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        pi: policy to evaluate
        V0: initial guess for V table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        V_real: real Value table
        learningRate: learning rate of the algorithm
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    
    # create a vector N to store the number of times each state has been visited
    N = np.copy(V0)
    # create a vector C to store the cumulative cost associated to each state
    C = np.zeros(env.nx)
    # create a vector V to store the Value
    V = np.zeros(env.nx)
    # create a list V_err to store history of the error between real and estimated V table
    V_err = []
    # for each episode
    for k in range(nEpisodes):
        # reset the environment to a random state
        x = env.reset()
        
        # simulate the system using the policy pi   
        for t in range(maxEpisodeLength):
            if callable(pi):        # pi is a function
                u = pi(env, x)
            else:                # pi is a vector
                u = pi[x]
            x_next, cost = env.step(u)
            V[x] += learningRate * (cost + gamma * V[x_next] - V[x])
            x = x_next

        # compute V_err as: mean(abs(V-V_real))
        V_err.append(np.mean(np.abs(V - V_real)))

        if(k % nprint == 0):
            print("TF(0) iter %d, V_err = %.6f" % (k, V_err[-1]))
            if(plot):
                env.plot_V_table(V)

    return V, V_err