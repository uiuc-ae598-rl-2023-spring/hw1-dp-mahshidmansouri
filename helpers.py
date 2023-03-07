# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:00:40 2023
AE 598 - RL HW1
@author: Mahshid Mansouri 
"""
import random
import numpy as np

# The following function calculates the value function for a specific state for all possible actions that could be taken from the state (i.e., one-step look ahead) 
def one_step_lookahead(env, state, V, gamma):
        
        """
        Args:
            
            state: A given state
            V: The value function vector of the size [num_states x 1] calculated using the policy evaluation algorithm 
             
        Returns:
            The vector A of the size [num_actions x 1] which is the expected value of all possible ctions that could be taken from the given input state 
        """
        A = np.zeros(env.num_actions)
        for action in range(env.num_actions):
            env.s = state                 
            next_state, reward , done = env.step(action)
            A[action] = (reward + gamma*V[next_state]) 
        return A
    

# Function to choose the next action greedly given the current state  for SARSA and Q_Learning algorithms
def epsilon_greedy(env, state, Q, epsilon):

    """  
    Calculates the greedy action given the current state and Q matrix
    Args:
        env: Environment
        state: The input state
        Q: A matrix of the size [num_states x num_actions] where each element is the action value function associated with a (state,action) pair 
        epsilon: greedy factor 
        
    Returns:
        Epsilon-greedy action 
    """
    
    # Calculate the greedy action
    greedy = np.argmax(Q[state,:])  
    
    # Calculate epsilon greedy probabilities
    weights = np.ones(env.num_actions)*epsilon/env.num_actions
    weights[greedy] = 1-epsilon+epsilon/env.num_actions

    # Choose action with epsilon greedy weights
    a_greedy = random.choices(range(env.num_actions), weights=weights, k=1)

    return a_greedy[0]



def epsilon_greedy_policy(env, Q):
    
    
    """  
    Calculates the epsilon-greedy policy for a fixded Q(s,a) without any sampling 
    Args:
        env: Environment
        Q: A matrix of the size [num_states x num_actions] where each element is the action value function associated with a (state,action) pair 
        
    Returns:
        Epsilon-greedy policy 
    """
    
    pi_star = np.zeros(env.num_states)
    
    # Loop over all states
    for state in range(env.num_states):
        
        # Identify greedy action
        greedy = np.argmax(Q[state][:])

        pi_star[state] = greedy

    return pi_star


# The following function learns the value function associated with the optimal policy produced by SARSA, which is a vector of size [num_states x 1]
def TD_zero(env, policy, total_episodes, alpha, gamma): 
    
    """  
     Args:
        env: The environment 
            env.num_states is a number of states in the environment. 
            env.num_actions is a number of actions in the environment.
            env.step(action) takes an action, and outputs is a list of transition tuples (next_state, reward, done).
            env.s sets the state of the environment to a specific states
        
        policy: Input policy which is a matrix of the dimension [num_states x num_actions] where each element in the matrix determines the probability of being in states s while performing action a, i.e., policy pi(a|s).
        gamma: Discount factor.
        alpha: Learning rate.
         

    Returns:
        The learned value function associated with the optimal policy produced by SARSA and Q-learning, which is a vector of size [num_states x 1]
    """
    
    V = np.zeros(env.num_states)
    #optimal_policy = np.where(policy == 1)[1]
    optimal_policy = policy
    for episode in range(total_episodes):
        #state = random.randrange(25)
        state = env.reset()
        while True:      
            ## Derive the optimal action for each state from the optimal policy matrix
            action = optimal_policy[state]
            next_state, reward, done = env.step(action)
            V[state] = V[state] + alpha*(reward + gamma*V[next_state]-V[state])
            state = next_state           
            if done: 
                break 
    return V
    




 
