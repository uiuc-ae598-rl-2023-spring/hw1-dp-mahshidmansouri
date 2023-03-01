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
    
    S = []
    AA = []
    R = []
    T = []
    time = 0
    r = 0
    
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
        
        env.s= state         
        r = env.step(best_action)[1]
        S.append(state)
        AA.append(best_action)
        R.append(r) 
        T.append(time/25)
        time += 1
        
    plt.plot(T, AA)
    plt.plot(T, S)
    plt.plot(T, R)
    plt.xlabel('Time (s)')
    plt.title('Trajectory for the trained agent')
    plt.legend(['Action','State','Reward'])
    
    return policy, V, mean_value_function, iter_num


############################################## Main ##############################################
# Create the environment
env = gridworld.GridWorld(hard_version=False)
gamma = 0.95
theta = 0.0001
policy, v, mean_value_function, iter_num = value_iteration(env, theta, gamma)


# Plot of the learning curve
plt.plot(iter_num, mean_value_function)
plt.xlabel("Number of Iterations")
plt.ylabel('Mean of the Value Function')
plt.title(r'Learning curve for policy iteration algorithm: $\theta$ = ' + str(theta))

# A plot of an example trajectory for each trained agent.
# A plot of the policy that corresponds to each trained agent.
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), (5,5)))
print("")

# A plot of the state-value function that corresponds to each trained agent 
print("Reshaped Grid Value Function:")
print(v.reshape((5,5)))
print("")

plt.plot(np.arange(25), v)
plt.xlabel("State")
plt.ylabel('V(s)')
plt.title("Learned Value Function")