#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def Greedy(Q, x, exploration_prob, nu):
    if uniform(0,1) < exploration_prob:
        return randint(nu)
    return np.argmin(Q[x,:])

def q_learning(env, gamma, Q, nEpisodes, maxEpisodeLength, 
               learningRate, exploration_prob, exploration_decreasing_decay,
               min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' Q-learning algorithm:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    # Keep track of the cost-to-go history (for plot)
    h_ctg = []
    # Make a copy of the initial Q table guess
    Q = Q.copy()
    # for every episode
    for ep in range (nEpisodes):
        # reset the state
        x = env.reset()
        # initialize cost-to-go for this episode
        cost_to_go = 0.0
        # simulate the system for maxEpisodeLength steps
        for t in range (maxEpisodeLength):
            # with probability exploration_prob take a random control input
            u = Greedy(Q, x, exploration_prob, env.nu)
            # otherwise take a greedy control
            x_next, cost = env.step(u)
            # accumulate cost-to-go
            cost_to_go += (gamma**t) * cost
            # Compute reference Q-value at state x
            Q_target = cost + gamma * min(Q[x_next, :])
            # Update Q-Table with the given learningRate
            Q[x, u] += learningRate * (Q_target - Q[x, u])
            # update state
            x = x_next

        # update epsilon with an exponential decay: eps = exp(-decay*episode)
        exploration_prob = np.exp(-exploration_decreasing_decay*ep)
        exploration_prob = max(exploration_prob, min_exploration_prob)  #saturate epsilon if it is below min_eps
        # keep track of the cost to go
        h_ctg.append(cost_to_go)
        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(plot and (ep % nprint == 0)):
            V, pi = compute_V_pi_from_Q(env, Q)
            env.plot_V_table(V)
            env.plot_policy(pi)
            print("Episode %d/%d - Average/min/max Value: %.2f / %.2f / %.2f" % 
                  (ep, nEpisodes, np.mean(V), np.min(V), np.max(V)) )
    
    return Q, h_ctg