import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import discrete_pendulum

## Functions Definitions
# Function to choose the next action greedly given the current state 
def choose_action(state,Q, epsilon):
    
    """  
    Args:
        state: The input state
        Q: A matrix of the size [num_states x num_actions] where each element is the action value function associated with a (state,action) pair 
        epsilon: greedy factor 
        
    Returns:
        Epsilon-greedy action 
    """

    action=0
    # Choose a random action 
    if np.random.uniform(0, 1) < epsilon:
        action = random.randrange(4)
    else:
        # Choose the action that maximizes the Q value for the given state
        action = np.argmax(Q[state, :])
    return action


# The following function chosses the actin that maximizes the Q function value associated with a given state
def Q_function_maximizier_action(state, Q):
    
    """  
    Args:
        state: The input state
        Q: A matrix of the size [num_states x num_actions] where each element is the action value function associated with a (state,action) pair 
        epsilon: greedy factor 
        
    Returns:
        The actin that maximizes the Q function value associated with a given state 
    """
    action = np.argmax(Q[state,:])  
    return action
    
# The following function is the update rule for the Q function for Q-Learning algorithm 
def update(state, next_state, reward, action, Q, alpha):
    
    """  
     Args:
        state: Current state
        next_state: Next state
        reward: Reward obtained
        action: Action that needs to be taken      
        alpha: Learning rate 
         

    Returns:
        Updates the Q function for the current state and action 
    """
    
    predict = Q[state, action]
    a_prime = Q_function_maximizier_action(next_state, Q)
    target = reward + gamma * np.max(Q[next_state, a_prime])
    Q[state, action] = Q[state, action] + alpha * (target - predict)
  
# The following function learns the value function associated with the optimal policy produced by Q-Learning, which is a vector of size [num_states x 1]
def TD_zero(env, policy, alpha, gamma): 
    
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
        num_episodes: The number of episodes
         

    Returns:
        The learned value function associated with the optimal policy produced by SARSA and Q-learning, which is a vector of size [num_states x 1]
    """
    
    V = np.zeros(env.num_states)
    optimal_policy = np.where(policy == 1)[1]
    for state in range(env.num_states):
        #state = random.randrange(25)
        env.s = state
        while True:      
            ## Derive the optimal action for each state from the optimal policy matrix
            action = optimal_policy[state]
            next_state, reward, done = env.step(action)
            V[state] = V[state] + alpha*(reward + gamma*(V[next_state]-V[state]))
            next_state = state          
            if done: 
                break 
    return V

def Q_Learning(env, epsilon, alpha, gamma, total_episodes, max_steps):
    
    """  
     Q-Learning Algorithm Implementation
     Args:
        env: The environment 
            env.num_states is a number of states in the environment. 
            env.num_actions is a number of actions in the environment.
            env.step(action) takes an action, and outputs is a list of transition tuples (next_state, reward, done).
            env.s sets the state of the environment to a specific states
        
        epsilon: Learning rate.
        gamma: Discount factor.
        alpha: Learning rate.
        total_episodes: The number of episodes.
        max_steps: The maximum number of steps taken within each episode.
         

    Returns:
        The learned optimal policy 
    """
    
    # Initializing the Q-matrix
    Q = np.zeros((env.num_states, env.num_actions))
    
    # Initializing the policy
    policy = np.zeros((env.num_states,env.num_actions))
    
    S = []
    A = []
    R = []
    T = []
    time = 0
    
    # Starting the SARSA learning
    for episode in range(total_episodes):     
        t = 0
        r = 0
        state = env.reset()
        action = choose_action(state, Q, epsilon)
     
        while t < max_steps:
             
            # Getting the next state
            next_state, reward, done = env.step(action)
     
            # Choosing the next action
            next_action = choose_action(next_state, Q, epsilon)
             
            # Learning the Q-value
            update(state, next_state, reward, action, Q, alpha)
            
            # Update the policy matrix when a (state,action) pair is seen. This policy will be used in TD(0) to learn the value function associated with the optimal policy.
            policy[state][action] = 1
     
            # Make the next state and action the current state and action
            state = next_state
            action = next_action
             
            # Updating the respective vaLues
            t += 1
            r = r + reward
            
            # Store the states, actions, rewards, and time 
            S.append(state)
            A.append(action)
            R.append(r)
            T.append(time)
            
            # If at the end of learning process
            if done: 
                break
            
    plt.plot(np.arange(max_steps),A[(total_episodes-1)*max_steps:total_episodes*max_steps])
    plt.plot(np.arange(max_steps),S[(total_episodes-1)*max_steps:total_episodes*max_steps])
    plt.plot(np.arange(max_steps),R[(total_episodes-1)*max_steps:total_episodes*max_steps])
    plt.xlabel('Time (s)')
    plt.title('Trajectory for the trained agent')
    plt.legend(['Action','State','Reward'])
    
    return policy
        

# Functions for plotting 
def learning_curve_plotter_for_different_alpha(epsilon, alpha):
    
    # Loop over different values of alpha, and plot the learning curve 
    for alpha in alpha:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the reward
        output_return = []
        episode_num = []
        policy = np.zeros((env.num_states,env.num_actions))
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0
            state = env.reset()
            action = choose_action(state, Q, epsilon)
         
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = choose_action(next_state, Q, epsilon)
                 
                # Learning the Q-value
                update(state, next_state, reward, action, Q, alpha)
                
                # Update the policy matrix when a (state,action) pair is seen. This policy will be used in TD(0) to learn the value function associated with the optimal policy.
                policy[state][action] = 1
         
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r = r + reward
                
                # If at the end of learning process
                if done:
                    output_return.append(r)
                    episode_num.append(episode)
                    break
    
        plt.plot(episode_num, output_return)
        plt.xlabel("number of episodes")
        plt.ylabel('Sum of rewards in each episode')
        plt.title(r'Return vs. number of episodes for Q-Learning algorithm')
        plt.legend([r'$\alpha$ = 0.1', r'$\alpha$ = 0.3', r'$\alpha$ = 0.5', r'$\alpha$ = 0.7', r'$\alpha$ = 0.9'])
    


def learning_curve_plotter_for_different_epsilon(epsilon, alpha):
    # Loop over different values of epsilon, and plot the learning curve 
    for epsilon in epsilon:
        
        # Initializing the Q-matrix
        Q = np.zeros((env.num_states, env.num_actions))
        
        # Initializing the reward
        output_return = []
        episode_num = []
        policy = np.zeros((env.num_states,env.num_actions))
        
        # Starting the SARSA learning
        for episode in range(total_episodes):     
            t = 0
            r = 0
            state = env.reset()
            action = choose_action(state, Q, epsilon)
         
            while t < max_steps:
                 
                # Getting the next state
                next_state, reward, done = env.step(action)
         
                # Choosing the next action
                next_action = choose_action(next_state, Q, epsilon)
                 
                # Learning the Q-value
                update(state, next_state, reward, action, Q, alpha)
                
                # Update the policy matrix when a (state,action) pair is seen. This policy will be used in TD(0) to learn the value function associated with the optimal policy.
                policy[state][action] = 1
         
                # Make the next state and action the current state and action
                state = next_state
                action = next_action
                 
                # Updating the respective vaLues
                t += 1
                r = r + reward
                
                # If at the end of learning process
                if done:
                    output_return.append(r)
                    episode_num.append(episode)
                    break
                
    
        plt.plot(episode_num, output_return)
        plt.xlabel("number of episodes")
        plt.ylabel('Sum of rewards in each episode')
        plt.title(r'Return vs. number of episodes for Q-Learning algorithm')
        plt.legend([r'$\epsilon$ = 0.1', r'$\epsilon$ = 0.3', r'$\epsilon$ = 0.5', r'$\epsilon$ = 0.7', r'$\epsilon$ = 0.9'])

    
############################################## Main ##############################################

# Parameters Initilization 
total_episodes = 100
max_steps = 100
gamma = 0.95

 
# Create the environment
## For the gridworld problem
#env = gridworld.GridWorld(hard_version=False)

## For the inveretd pendulum problem 
n_theta, n_thetadot, n_tau = 5, 5 ,5
env =discrete_pendulum.Pendulum(n_theta, n_thetadot, n_tau)

# A plot of learning curves for several different values of epsilon
plt.figure(1)
learning_curve_plotter_for_different_epsilon(epsilon = [0.1, 0.3, 0.5, 0.7, 0.9], alpha = 0.95)
    
# A plot of learning curves for several different values of alpha
plt.figure(2)
learning_curve_plotter_for_different_alpha(epsilon = 0.1, alpha = [0.1, 0.3, 0.5, 0.7, 0.9])

#A plot of an example trajectory for each trained agent.
# A plot of the policy that corresponds to each trained agent.
alpha = 0.95
epsilon = 0.1
policy = Q_Learning(env, epsilon, alpha, gamma, total_episodes, max_steps)

print("Reshaped Grid Policy:")
print(np.reshape(np.argmax(policy, axis=1), (int(np.sqrt(env.num_states)),int(np.sqrt(env.num_states)))))
print("")
        

# A plot of the state-value function that corresponds to each trained agent
V = TD_zero(env, policy, alpha, gamma)
print("Reshaped TD(0)-Learned State Value Function :")
print(V.reshape(5,5))

# For the Gridworld
plt.plot(np.arange(25), V)
plt.xlabel("State")
plt.ylabel('V(s)')
plt.title("Learned Value Function")

