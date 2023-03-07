"""
Created on Mon Mar  6 11:00:40 2023
AE 598 - RL HW1
@author: Mahshid Mansouri 
"""

import numpy as np
import matplotlib.pyplot as plt

import discrete_pendulum

from HW1_SARSA import SARSA
from HW1_Q_Learning import Q_Learning
from HW1_TD_Zero import TD_zero 


from plotters import  plot_example_traj_pendulum
from plotters import SARSA_learning_curve_plotter_for_different_alpha
from plotters import SARSA_learning_curve_plotter_for_different_epsilon
from plotters import Q_learning_curve_plotter_for_different_alpha
from plotters import Q_learning_curve_plotter_for_different_epsilon


# Create the environment
n_theta, n_thetadot, n_tau = 20, 20 ,20
env =discrete_pendulum.Pendulum(n_theta, n_thetadot, n_tau)

############################################## SARSA ##############################################

def SARSA_main():
    
    # Parameters Initilization 
    total_episodes = 10000
    max_steps = 1000
    gamma = 0.95

    epsilon = 0.9
    alpha = 0.5
    
    # Plot of the learning curve
    plt.figure(1)
    Q, policy = SARSA(env, epsilon, alpha, gamma, total_episodes, max_steps)
    pi_star = policy 
    
    # A plot of the policy that corresponds to each trained agent.
    plt.figure(2)
    
    # A plot of learning curves for several different values of epsilon
    plt.figure(3)
    SARSA_learning_curve_plotter_for_different_epsilon(epsilon = [0.1, 0.3, 0.5, 0.7, 0.9], alpha = 0.5)
        
    # A plot of learning curves for several different values of alpha
    plt.figure(4)
    SARSA_learning_curve_plotter_for_different_alpha(epsilon = 0.9, alpha = [0.1, 0.3, 0.5, 0.7, 0.9])
                  
    # A plot of an example trajectory for each trained agent.
    plt.figure(5)
    pi_star = policy
    plot_example_traj_pendulum(pi_star, env)
      
    # A plot of the state-value function that corresponds to each trained agent
    plt.figure(6)
    V = TD_zero(env, policy, total_episodes, alpha, gamma)
    plt.plot(np.arange(25),V)
    plt.ylabel("V(s)")
    plt.xlabel("State")
    plt.title(r'Learned value function: $\epsilon$ = ' + str(epsilon) + r', $\alpha$ = '+ str(alpha))
        

############################################## Q_Learning ##############################################

def Q_Learning_main():
    
    # Parameters Initilization 
    total_episodes = 10000
    max_steps = 1000
    gamma = 0.95

    epsilon = 0.9
    alpha = 0.5
    
    # Plot of the learning curve
    plt.figure(1)
    Q, policy = Q_Learning(env, epsilon, alpha, gamma, total_episodes, max_steps)
    pi_star = policy 
    
    # A plot of the policy that corresponds to each trained agent.
    plt.figure(2)
    
    # A plot of learning curves for several different values of epsilon
    plt.figure(3)
    Q_learning_curve_plotter_for_different_epsilon(epsilon = [0.1, 0.3, 0.5, 0.7, 0.9], alpha = 0.5)
        
    # A plot of learning curves for several different values of alpha
    plt.figure(4)
    Q_learning_curve_plotter_for_different_alpha(epsilon = 0.9, alpha = [0.1, 0.3, 0.5, 0.7, 0.9])
                  
    # A plot of an example trajectory for each trained agent.
    plt.figure(5)
    pi_star = policy
    plot_example_traj_pendulum(pi_star, env)
      
    # A plot of the state-value function that corresponds to each trained agent
    plt.figure(6)
    V = TD_zero(env, policy, total_episodes, alpha, gamma)
    plt.plot(np.arange(25),V)
    plt.ylabel("V(s)")
    plt.xlabel("State")
    plt.title(r'Learned value function: $\epsilon$ = ' + str(epsilon) + r', $\alpha$ = '+ str(alpha))
        
