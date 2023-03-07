# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:00:40 2023
AE 598 - RL HW1
@author: Mahshid Mansouri 
"""

import numpy as np
import matplotlib.pyplot as plt
from helpers import epsilon_greedy
from helpers import epsilon_greedy_policy
from HW1_SARSA import SARSA_update
from HW1_Q_Learning import Q_update


def plot_gridworld_policy(pi_star):
    """
    Plots gridworld policy
    :param pi_star: policy to plot
    :return: figure
    """

    row = 10
    pi_star = np.squeeze(pi_star)
    fig, ax = plt.subplots(1,1)

    counter = 0
    for i in range(5):  # row
        col = 0
        for j in range(5):  # col
            pi = pi_star[counter]
            # get coords

            if pi == 0:  # right
                x = col+0.5
                y = row-1
                dx = 1
                dy = 0
            elif pi == 2:  # left
                x = col+1.5
                y = row-1
                dx = -1
                dy = 0
            elif pi == 1:  # up
                x = col+1
                y = row-1.5
                dx = 0
                dy = 1
            elif pi == 3:  # down
                x = col+1
                y = row-0.5
                dx = 0
                dy = -1

            plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, head_length=0.2, color='k')
            counter = counter + 1
            col = col + 2
        row = row - 2

    ax.set_xticks(np.arange(0, 10, 2))
    ax.set_yticks(np.arange(0, 10, 2))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, 10])
    plt.ylim([0,10])
    plt.grid()
    plt.title('Gridworld Policy')
    return

def plot_example_traj_grid(pi_star, env):
    """
    Plots gridworld example traj
    :param pi_star: optimal policy
    :param env: environment object
    :return: figure
    """
    
    # Initialize simulation
    s = env.reset()
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    pi_star = np.squeeze(pi_star)

    # Simulate until episode is done
    done = False
    while not done:
        a = pi_star[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    ax1 = plt.subplot()
    ax1.plot(log['t'], log['s'])
    ax1.plot(log['t'][:-1], log['a'])
    ax1.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
               
    return

def plot_example_traj_pendulum(pi_star, env):
    """
    Plots example trajectory of pendulum
    :param pi_star: optimal policy to run for epispode
    :param env: environment object
    :return: figure
    """
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],  # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],  # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = pi_star[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    # Clip theta
    thetas = []
    for theta in log['theta']:
        thetas.append(((theta + np.pi) % (2 * np.pi)) - np.pi)


    # Plot data and save to png file
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    ax[0].plot(log['t'], log['s'])
    ax[0].plot(log['t'][:-1], log['a'])
    ax[0].plot(log['t'][:-1], log['r'])
    ax[0].legend(['s', 'a', 'r'])
    ax[1].plot(log['t'], thetas)
    ax[1].plot(log['t'], log['thetadot'])
    ax[1].legend(['theta', 'thetadot'])
    
    return

def SARSA_learning_curve_plotter_for_different_alpha(env, epsilon, alpha, total_episodes, max_steps, gamma):
    
    # Loop over different values of alpha, and plot the learning curve 
    for alpha in alpha:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the policy
        policy = np.zeros((env.num_states,env.num_actions))
        output_return = []
        episode_num = []
        
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0 
            counter = 0
            state = env.reset()
            
            action = epsilon_greedy(env, state, Q, epsilon)
            
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = epsilon_greedy(env, next_state, Q, epsilon)
                 
                # Learning the Q-value
                SARSA_update(state, next_state, reward, action, next_action, Q, alpha, gamma)
                
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r += gamma**counter * reward
                counter += 1
    
                # If at the end of learning process
                if done:             
                    break
            output_return.append(r)
            episode_num.append(episode)
            
       
        plt.plot(episode_num, output_return)
        plt.xlabel("Number of episodes")
        plt.ylabel('Return')
        #plt.title(r'Return vs. number of episodes for SARSA algorithm')
        plt.legend([r'$\alpha$ = 0.1', r'$\alpha$ = 0.3', r'$\alpha$ = 0.5', r'$\alpha$ = 0.7', r'$\alpha$ = 0.9'])
    

def SARSA_learning_curve_plotter_for_different_epsilon(env, epsilon, alpha, total_episodes, max_steps, gamma):
    
    # Loop over different values of alpha, and plot the learning curve 
    for epsilon in epsilon:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the policy
        policy = np.zeros((env.num_states,env.num_actions))
        output_return = []
        episode_num = []
        
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0 
            counter = 0
            state = env.reset()
            
            action = epsilon_greedy(env, state, Q, epsilon)
            
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = epsilon_greedy(env, next_state, Q, epsilon)
                 
                # Learning the Q-value
                SARSA_update(state, next_state, reward, action, next_action, Q, alpha, gamma)
                
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r += gamma**counter * reward
                counter += 1
    
                # If at the end of learning process
                if done:             
                    break
            output_return.append(r)
            episode_num.append(episode)
        
        plt.plot(episode_num, output_return)
        plt.xlabel("Number of episodes")
        plt.ylabel('Return')
        #plt.title(r'Return vs. number of episodes for SARSA algorithm')
        plt.legend([r'$\epsilon$ = 0.1', r'$\epsilon$ = 0.3', r'$\epsilon$ = 0.5', r'$\epsilon$ = 0.7', r'$\epsilon$ = 0.9'])

def Q_learning_curve_plotter_for_different_alpha(env, epsilon, alpha, total_episodes, max_steps, gamma):
    
    # Loop over different values of alpha, and plot the learning curve 
    for alpha in alpha:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the policy
        policy = np.zeros((env.num_states,env.num_actions))
        output_return = []
        episode_num = []
        
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0 
            counter = 0
            state = env.reset()
            
            action = epsilon_greedy(env, state, Q, epsilon)
            
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = epsilon_greedy(env, next_state, Q, epsilon)
                 
                # Learning the Q-value
                Q_update(state, next_state, reward, action, Q, alpha, gamma)
                
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r += gamma**counter * reward
                counter += 1
    
                # If at the end of learning process
                if done:             
                    break
            output_return.append(r)
            episode_num.append(episode)
            
       
        plt.plot(episode_num, output_return)
        plt.xlabel("Number of episodes")
        plt.ylabel('Return')
        #plt.title(r'Return vs. number of episodes for Q-Learning algorithm')
        plt.legend([r'$\alpha$ = 0.1', r'$\alpha$ = 0.3', r'$\alpha$ = 0.5', r'$\alpha$ = 0.7', r'$\alpha$ = 0.9'])
    

def Q_learning_curve_plotter_for_different_epsilon(env, epsilon, alpha, total_episodes, max_steps, gamma):
    
    # Loop over different values of alpha, and plot the learning curve 
    for epsilon in epsilon:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the policy
        policy = np.zeros((env.num_states,env.num_actions))
        output_return = []
        episode_num = []
        
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0 
            counter = 0
            state = env.reset()
            
            action = epsilon_greedy(env, state, Q, epsilon)
            
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = epsilon_greedy(env, next_state, Q, epsilon)
                 
                # Learning the Q-value
                Q_update(state, next_state, reward, action, Q, alpha, gamma)
                
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r += gamma**counter * reward
                counter += 1
    
                # If at the end of learning process
                if done:             
                    break
            output_return.append(r)
            episode_num.append(episode)
        
        plt.plot(episode_num, output_return)
        plt.xlabel("Number of episodes")
        plt.ylabel('Return')
        #plt.title(r'Return vs. number of episodes for Q-Learning algorithm')
        plt.legend([r'$\epsilon$ = 0.1', r'$\epsilon$ = 0.3', r'$\epsilon$ = 0.5', r'$\epsilon$ = 0.7', r'$\epsilon$ = 0.9'])
