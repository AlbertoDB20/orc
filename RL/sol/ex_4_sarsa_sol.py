#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def epsGreedy(Q, x, eps, nu):
    if uniform(0,1) < eps:
        return randint(nu)
    return np.argmin(Q[x,:])

def sarsa(env, gamma, Q, nEpisodes, maxEpisodeLength, 
          learningRate, eps, eps_decreasing_decay, min_eps, 
          compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' SARSA:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        eps: initial exploration probability for epsilon-greedy policy
        eps_decreasing_decay: rate of exponential decay of epsilon
        min_eps: lower bound of epsilon
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = []           # Learning history (for plots)
    # for every episode
    for ep in range (nEpisodes):
        # reset the environment state
        x = env.reset()
        # compute u with an epsilon-greedy policy
        u = epsGreedy(Q, x, eps, env.nu)
        # initialize cost-to-go for this episode
        J = 0.0
        # for every step of the episode
        for t in range (maxEpisodeLength):
            # take action u, observe x_next, c
            x_next, cost = env.step(u)
            # compute u_next with an epsilon-greedy policy
            u_next = epsGreedy(Q, x_next, eps, env.nu)
            # compute Q target
            Q_target = cost + gamma * Q[x_next, u_next]
            # update Q function with TD
            Q[x,u] += learningRate * (Q_target - Q[x,u])
            # update x, u with x_next, u_next
            x, u = x_next, u_next
            # update cost-to-go for this episode
            J += (gamma**t) * cost

        # update epsilon with exponentially decaying function: eps=e^(-decay*episode)
        eps = np.exp(-eps_decreasing_decay*ep)
        eps = max(eps, min_eps)  #saturate epsilon if it is below min_eps

        # append cost-to-go to list h_ctg (for plots)
        h_ctg.append(J)
        # every nprint episodes print mean V, mean cost-to-go, and epsilon
        if (ep % nprint == 0):
            V, pi = compute_V_pi_from_Q(env, Q)
            print("Episode %d/%d - Mean V: %.2f - Mean cost-to-go: %.2f - Epsilon: %.4f" 
                  % (ep, nEpisodes, np.mean(V), np.mean(h_ctg[-nprint:]), eps*100))
            if plot:
                env.plot_V(V, ep)
        
    return Q, h_ctg