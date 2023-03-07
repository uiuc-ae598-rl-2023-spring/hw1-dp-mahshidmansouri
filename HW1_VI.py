"""
Created on Mon Mar  6 11:00:40 2023
AE 598 - RL HW1
@author: Mahshid Mansouri 
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

## Functions Definitions
# The following function calculates the expected value for all actions in a given state.
def one_step_lookahead(env, state, V):
        
        """
        Args:
            
            state: A given state
            V: The value function vector of the size [num_states x 1] calculated using the policy evaluation algorithm 
             
        Returns:
            The vector A of the size [num_actions x 1] which is the expected value of all possible ctions that could be taken from the given input state 
        """
        
        # A vector of length env.n containing the expected value of each action.
        A = np.zeros(env.num_actions)
        for action in range(env.num_actions):      
            env.s = state                 
            next_state, reward , done = env.step(action)
            A[action] = reward + gamma*V[next_state] 
        return A


def value_iteration(env, theta, gamma):
    
    """
    Value Iteration Algorithm.
    
    Args:
        env: The environment 
            env.num_states is a number of states in the environment. 
            env.num_actions is a number of actions in the environment.
            env.step(action) takes an action, and outputs is a list of transition tuples (next_state, reward, done).
            env.s sets the state of the environment to a specific states
        
        gamma: discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.
         
    Returns:
        A tuple (policy, V, mean_value_function, iter_num) which are the othe value function associated with the policy, the mean of the final value function for all states, and the number of iterations of the algorithm.
    """
   
    mean_value_function = []
    iter_num = []
    
    V = np.zeros(env.num_states)
    value_iter_num = 0
    
    while True:
        iter_num.append(value_iter_num)
        mean_value_function.append(np.mean(V))
        # Stopping condition
        delta = 0
        # Update each state...
        for state in range(env.num_states):
            
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(env, state, V)
            best_action_value = np.max(A)
            
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[state]))
            
            # Update the value function.
            V[state] = best_action_value     

        value_iter_num +=1
        
        # Check if we can stop 
        if delta < theta:

            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.num_states, env.num_actions])
    
    for state in range(env.num_states):
        
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(env, state, V)
        best_action = np.argmax(A)
        
        # Always take the best action
        policy[state, best_action] = 1
    
    return policy, V, mean_value_function, iter_num

def plot_gridworld_policy(pi_star):
    """
    Plots gridworld policy
    :param pi_star: policy to plot
    :param name: figure name
    :param modelFreeTag: True if model-free, False if model-based
    :param logPass: extra data if model-free
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
        s, r, done = env.step(a)
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
############################################## Main ##############################################

# # Create the environment
# env = gridworld.GridWorld(hard_version=False)
# gamma = 0.95
# theta = 0.0001
# policy, v, mean_value_function, iter_num = value_iteration(env, theta, gamma)


# # Plot of the learning curve
# plt.figure(1)
# plt.plot(iter_num, mean_value_function)
# plt.xlabel("Number of Iterations")
# plt.ylabel('Mean of the Value Function')
# plt.title(r'Learning curve plot: $\theta$ = ' + str(theta))

# # A plot of an example trajectory for each trained agent.
# plt.figure(2)
# pi_star = (np.reshape(np.argmax(policy, axis=1), (25,1)))
# plot_example_traj_grid(pi_star, env)

# # A plot of the policy that corresponds to each trained agent.
# plt.figure(3)
# plot_gridworld_policy(pi_star)

# #A plot of the state-value function that corresponds to each trained agent
# plt.figure(4)
# plt.plot(np.arange(25),v)
# plt.xlabel("State(s)")
# plt.ylabel('V(s)')
# plt.title(r'Learned value function: $\theta$ = ' + str(theta))