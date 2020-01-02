#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: The Taxi Problem
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model 
# 
# Consider once again the taxi domain described in the Homework which you modeled using a Markov decision process. In this lab you will interact with larger version of the same problem. You will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem. The domain is represented in the diagram below.
# 
# <img src="taxi.png" width="200px">
# 
# In the taxi domain above,
# 
# * The taxi can be in any of the 25 cells in the diagram. The passenger can be at any of the 4 marked locations ($Y$, $B$, $G$, $R$) or in the taxi. Additionally, the passenger wishes to go to one of the 4 possible destinations. The total number of states, in this case, is $25\times 5\times 4$.
# * At each step, the agent (taxi driver) may move in any of the four directions -- south, north, east and west. It can also pickup the passenger or drop off the passenger. 
# * The goal of the taxi driver is to pickup the passenger and drop it at the passenger's desired destination.
# 
# **Throughout the lab, use $\gamma=0.99$.**
# 
# $$\diamond$$

# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.
# 
# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `S` that contains all the states in the MDP. There is a total of $501$ states describing the possible taxi-passenger configurations. Those states are represented as strings of the form `"(x, y, p, d)"`, where $(x,y)$ represents the position of the taxi in the grid, $p$ represents the position of the passenger ($R$, $G$, $Y$, $B$, or in the taxi), and $d$ the destination of the passenger ($R$, $G$, $Y$, $B$). There is one additional absorbing state called `"Final"` to which the MDP transitions after reaching the goal.
# * An array `A` that contains all the actions in the MDP. Each action is represented as a string `"South"`, `"North"`, and so on.
# * An array `P` containing 5 $501\times 501$ sub-arrays, each corresponding to the transition probability matrix for one action.
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(S, A, (Pa, a = 0, ..., 5), c, g)`, where `S` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with 6 elements, where `P[a]` is an np.array corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# **Note**: Don't forget to import `numpy`.
# 
# ---

# In[7]:


# Add your code here.
import numpy as np
import numpy.random as rand

def load_mdp(filename, gamma):
    info = np.load(filename)
    S = info['S.npy']
    A = info['A.npy']
    Pa = info['P.npy']
    P = ()
    for probs in Pa:
        P += (probs,)
    c = info['c.npy']
    g = gamma
    return (S, A, P, c, g)


# We provide below an example of application of the function with the file `taxi.npz` that you can use as a first "sanity check" for your code.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('taxi.npz', 0.99)
# 
# rand.seed(42)
# 
# # States
# print('Number of states:', len(M[0]))
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('Random state:', M[0])
# 
# # Final state
# print('Final state:', M[0][-1])
# 
# # Actions
# print('Number of actions:', len(M[1]))
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('Random action:', M[1][a])
# 
# # Transition probabilities
# print('Transition probabilities for the selected state/action:')
# print(M[2][a][s, :])
# 
# # Cost
# print('Cost for the selected state/action:')
# print(M[3][s, a])
# 
# # Discount
# print('Discount:', M[4])
# ```
# 
# Output:
# 
# ```
# Number of states: 501
# Random state: (1, 0, 0, 2)
# Final state: Final
# Number of actions: 6
# Random action: West
# Transition probabilities for the selected state/action:
# [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
#   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# Cost for the selected state/action:
# 0.7
# Discount: 0.99
# ```

# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# You will now describe the policy that, at each state $x$, always moves the taxi down (South). Recall that the action "South" corresponds to the action index $0$. Your policy should be a `numpy` array named `pol` with as many rows as states and as many columns as actions, where `pol[s,a]` should contain the probability of action `a` in state `s` according to the desired policy. 
# 
# ---

# In[8]:



# Add your code here
M = load_mdp('taxi.npz', 0.99)
down = np.ones((len(M[0]),1))
actions = np.zeros((len(M[0]),len(M[1])-1)) 
pol = np.append(down,actions,axis=1)


# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy.
# 
# ---

# In[10]:


# Add your code here.
def evaluate_pol(m,plc):
    ppol = m[2][0]*plc[:,0]
    cpol = m[3][:,0]*plc[:,0]
    print(len(m[0]))
    for i in range(1,len(m[2])):
        ppol += m[2][i]*plc[:,i]
        cpol += m[3][:,i]*plc[:,i]
    return np.linalg.inv((np.identity(len(m[0])))-0.99*ppol).dot(cpol.reshape(-1,1))


# As an example, you can evaluate the policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jpi = evaluate_pol(M, pol)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jpi[s])
# ```
# 
# Output: 
# ```
# Cost to go at state (1, 0, 0, 2): [70.]
# Cost to go at state (4, 1, 3, 3): [70.]
# Cost to go at state (3, 2, 2, 0): [70.]
# ```

# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# ---

# In[12]:


# Add your code here.
import time

def value_iteration(m):
    J = np.zeros((len(m[0]),1))
    err = 1
    i = 0
    t_start = time.time()
    Q = []
    for j in range(len(m[1])):
        Q.append([])
    while err > 1e-8:
        
        for j in range(len(m[1])):
            Q[j] = m[3][:,j,None] + 0.99 * (m[2][j]).dot(J) 
        Jnew = np.min((Q),axis = 0)
        err = np.linalg.norm(Jnew - J)
        i += 1 
        J = Jnew
    t_end = time.time() - t_start
    print('Execution time: ', round(t_end,3))
    print('N. interations: ', i)
    return J


# For example, the optimal cost-to-go for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# Jopt = value_iteration(M)
# 
# rand.seed(42)
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = rand.randint(len(M[0]))
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.031 seconds
# N. iterations: 18
# Cost to go at state (1, 0, 0, 2): [4.1]
# Cost to go at state (4, 1, 3, 3): [4.76]
# Cost to go at state (3, 2, 2, 0): [6.69]
# 
# Is the policy from Activity 2 optimal? False
# ```

# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Your function should print the time it takes to run before returning, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[13]:


# Add your code here.

def policy_iteration(mdp):
    pi = np.ones(((len(mdp[0])), len(mdp[1]))) / len(mdp[1]) #shape (states, actios)
    quit = False
    Q = []
    for j in range(len(mdp[1])):
        Q.append([])
    iters = 0
    t = time.time()
    while not quit:
        cpi = np.diag(pi[:,0]).dot(mdp[3][:,0, None])
        Ppi = np.diag(pi[:,0]).dot(mdp[2][0])
        for i in range(1,len(mdp[1])):
            cpi += np.diag(pi[:,i]).dot(mdp[3][:,i, None])
            Ppi += np.diag(pi[:,i]).dot(mdp[2][i])
        J =np.linalg.inv( np.eye(len(mdp[0]))  - mdp[4] * Ppi).dot(cpi)
        for j in range(len(mdp[1])):
            Q[j] = mdp[3][:,j,None] + mdp[4] * (mdp[2][j]).dot(J)
        pinew = np.zeros(((len(mdp[0])), len(mdp[1])))
        for j in range(len(mdp[1])):
            pinew[:,j,None] = np.isclose(Q[j], np.min(Q, axis=0), atol=1e-8, rtol=1e-8).astype(int)
        pinew = pinew / np.sum(pinew, axis=1, keepdims = True)

        quit = (pi == pinew).all()
        pi = pinew
        iters += 1
    print("Execution time: ", round(time.time() - t,3))
    print("N. iterations: ", iters)
    return pi


# For example, the optimal policy for the MDP from **Activity 1** is can be computed as follows.
# 
# ```python
# popt = policy_iteration(M)
# 
# rand.seed(42)
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = rand.randint(len(M[0]))
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# ```
# 
# Output:
# ```
# Execution time: 0.089 seconds
# N. iterations: 3
# Policy at state (1, 0, 0, 2): North
# Policy at state (2, 3, 2, 2): West
# Policy at state (1, 4, 2, 0): West
# ```

# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, corresponding to a state index
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **100** trajectories of 10,000 steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair â˜ºï¸.
# 
# ---

# In[14]:


# Add your code here.
def simulate(m, plc, state):
    c = 0
    print(len(m[2]))
    for traj in range(100):
        for t in range(10000):
            a = np.random.choice(len(m[1]), p=plc[state])
            next_state = np.random.choice(len(m[0]), p=m[2][a][state])
            discnt = m[4]**t
            c += m[3][state, a]*discnt
            state = next_state
    return c


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# 
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# 
# # Select random state, and evaluate for the optimal policy
# s = rand.randint(len(M[0]))
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', Jopt[s])
# print('\tEmpirical:', simulate(M, popt, s))
# ```
# 
# Output:
# ````
# Cost-to-go for state (1, 0, 0, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.39338954193
# Cost-to-go for state (3, 1, 4, 1):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.09638954193
# Cost-to-go for state (3, 2, 2, 2):
# 	Theoretical: [ 4.1]
# 	Empirical: 4.3816865569
# ```

# In[ ]:




