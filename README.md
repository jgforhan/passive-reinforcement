# Passive Agent

## Environment
The environment is an $N\times{}M$ grid  
Unpathable obstacles can be placed in within this grid  
There are two goal states: one has a positive reward while the other has a negative reward

## Model
The agent learns about its environment using passive adaptive dynamic programming (passive ADP)  
Since the agent performs passive reinforcement learning, it follows a fixed policy to determine utilities of states it encounters  
Utilities are approximated at each step using the value iteration algorithm, over the fixed policy

## Design
Descriptive constants and helper funtctions used throughout to aid readability of code  
Parameters for both world and agent can be modified to adapt agent to new rulesets and environments  
Data structures chosen to improve running time and storage
