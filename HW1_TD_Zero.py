"""
Created on Mon Mar  6 11:00:40 2023
AE 598 - RL HW1
@author: Mahshid Mansouri 
"""

import numpy as np

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
    
    
    