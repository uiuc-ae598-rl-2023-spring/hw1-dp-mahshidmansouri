import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld

## Functions Definitions
# The following function calculates the value function associated with a given policy
def policy_eval(env,policy, gamma, theta): 
    
    """
    Policy Evaluation Algorithm.
    
    Args:
        env: The environment 
            env.num_states is a number of states in the environment. 
            env.num_actions is a number of actions in the environment.
            env.step(action) takes an action, and outputs is a list of transition tuples (next_state, reward, done).
            env.s sets the state of the environment to a specific states
        
        policy: Input policy which is a matrix of the dimension [num_states x num_actions] where each element in the matrix determines the probability of being in states s while performing action a, i.e., policy pi(a|s).
        gamma: Discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.
         
    Returns:
        A tuple (V, mean_value_function, iter_num) which are the value function associated with the policy, the mean of the final value function for all states, and the number of iterations of the algorithm.
    """
    
    V = np.zeros(env.num_states)
    policy_iter_num = 0
    mean_value_function = []
    iter_num = []
    while True:
        delta = 0
        iter_num.append(policy_iter_num)
        mean_value_function.append(np.mean(V))
        
        # For each state, perform a "full backup"
        for state in range(env.num_states):
            v = 0
            
            # Look at the possible next actions
            for action, action_prob in enumerate(policy[state]):
                
                # For each action, look at the possible next states...                
                # Calculate the expected value
                    env.s = state
                    next_state, reward, done = env.step(action)
                    v += 0.25* (reward + gamma * V[next_state])
                    
            # Calculate how much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[state]))
            V[state] = v
        policy_iter_num += 1 
        
        # Stop evaluating once our value function change is below a threshold theta
        if delta < theta:
            break
    return np.array(V), mean_value_function, iter_num 



# The following function calculates the value function for a specific state for all possible actions that could be taken from the state (i.e., one-step look ahead) 
def one_step_lookahead(env, state, V):
        
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
            A[action] = reward + gamma*V[next_state] 
        return A

# The following function implements the policy improvement algorithm 
def policy_improvement(env, gamma, theta): 
        
    """
    Policy Improvement Algorithm.
    
    Args:
        env: The environment 
            env.num_states is a number of states in the environment. 
            env.num_actions is a number of actions in the environment.
            env.step(action) takes an action, and outputs is a list of transition tuples (next_state, reward, done).
            env.s sets the state of the environment to a specific states
        
        gamma: discount factor.
        theta: We stop evaluation once our value function change is less than theta for all states.
         
    Returns:
        A tuple (policy, V, mean_value_function, iter_num ) which are the optimal policy, value function associated with the optimal policy, the mean of the optimal value function for all states, and the number of iterations of the algorithm.
    """
    
    # Start with a random policy
    policy = np.ones([env.num_states, env.num_actions])/env.num_actions
   
    while True:
        
        # Evaluate the current policy
        V, mean_value_function, iter_num = policy_eval(env, policy, gamma, theta)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state, perform the following ...
        for state in range(env.num_states):
            
            # The best action we would take under the current policy
            chosen_a = np.argmax(policy[state])
            
            # Find the best action by one-step lookahead
            action_values = one_step_lookahead(env, state, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[state] = np.eye(env.num_actions)[best_a]/env.num_actions
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V, mean_value_function, iter_num 

############################################## Main ##############################################
# Create the environment
env = gridworld.GridWorld(hard_version=False)

# Parameters Initialization
gamma = 0.95 
theta = 0.001
policy, v, mean_value_function, iter_num  = policy_improvement(env, gamma, theta)


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

    
    


