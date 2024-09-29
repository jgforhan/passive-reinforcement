import numpy as np
import transitions as tr
import random
from copy import copy
from consts import *

class Agent:
    ## Prob of random turn
    DISCOUNT_FACTOR = 1.0
    LEFT_PROB = 0.1
    RIGHT_PROB = 0.1
    DIRS = 4
    def __init__(self, world, reward, policy):
        self.rows = len(world)
        self.cols = len(world[0])

        ## current position
        self.posn = [0,0]
        self.world_rep = world
        self.reward = reward
        self.policy = policy
        self.values = np.array([[0.0 for col in range(self.cols)] for row in range(self.rows)])

        ##TODO: automate from world info
        self.values[1][3] = -1.0
        self.values[2][3] = 1.0
        
        ## structures for transition probabilities
        self.num_tr_generic = np.array([[[0 for dir in range(self.DIRS)] for col in range(self.cols)] for row in range(self.rows)])
        self.num_tr_specific = np.array([[tr.TransitionMap() for col in range(self.cols)] for row in range(self.rows)])
        self.tr_prob = np.array([[tr.TransitionMap() for col in range(self.cols)] for row in range(self.rows)])

    def get_posn(self):
        return self.posn
    
    def get_world(self):
        return self.world_rep

    def get_reward_fn(self):
        return self.reward

    def get_policy(self):
        return self.policy

    def get_expected_utils(self):
        return self.values
    
    ## Potentially make random alteration to move
    def __alter_move__(self, intent, rand):
        action = None
        if intent == UP:
            if rand < self.LEFT_PROB:
                action = LEFT
            elif rand > (1 - self.RIGHT_PROB):
                action = RIGHT
            else:
                action = UP
        elif intent == DOWN:
            if rand < self.LEFT_PROB:
                action = RIGHT
            elif rand > (1 - self.RIGHT_PROB): 
                action = LEFT
            else:
                action = DOWN
        elif intent == LEFT:
            if rand < self.LEFT_PROB:
                action = DOWN
            elif rand > (1 - self.RIGHT_PROB):
                action = UP
            else:
                action = LEFT
        elif intent == RIGHT:
            if rand < self.LEFT_PROB:
                action = UP
            elif rand > (1 - self.RIGHT_PROB):
                action = DOWN
            else:
                action = RIGHT
        return action

    ## Make a move and check if it is valid
    ## If it is, return True, else return False
    def __validate_move__(self, temp_posn, action):
        ## apply action to temp_posn
        if action == UP:
            temp_posn[0] -= 1
        elif action == DOWN:
            temp_posn[0] += 1
        elif action == LEFT:
            temp_posn[1] -= 1
        else:
            temp_posn[1] += 1

        ## Check if new move is within bounds of the world
        ## If it is not then return False
        if temp_posn[0] < 0 or temp_posn[0] >= self.rows:
            return False
        if temp_posn[1] < 0 or temp_posn[1] >= self.cols:
            return False
        if self.world_rep[temp_posn[0]][temp_posn[1]] == NO_PATH:
            return False
        return True
    
    ## Choose a move based on the policy
    ## Potentially alter the intended move randomly
    ## Make the move if it is valid (updates self.posn on call)
    ## Return the intended action
    def __select_move__(self):
        ## Get intent and generate random number to decide if we alter direction
        intent = self.policy[self.posn[0]][self.posn[1]]
        rand = random.random()
        temp_posn = copy(self.posn)
        
        ## Intent is chosen based on policy, but there is random chance
        ## of the agent turning left or right 90deg from its perspective
        action = self.__alter_move__(intent, rand)

        ## Checks if new state is legal
        if self.__validate_move__(temp_posn, action):
            ## if the new position is valid, update it
            self.posn = temp_posn
        ## Return intended action
        return intent
        
    def __update_tr__(self, init, fin, action):
        ## Increment transition from initial state using action
        self.num_tr_generic[init[0]][init[1]][action] += 1

        ## Increment transition from initial state to final state using action
        ## key is given by [new_row, new_col, action_used]
        self.num_tr_specific[init[0]][init[1]].increment_transition(key=(fin[0], fin[1], action))

        ## Consider keys for the initial state
        for key in self.num_tr_specific[init[0]][init[1]].get_dictionary():
            ## Consider the keys that use the performed action
            if key[ACTION_KEY] == action:
                ## Get empirical conditional probability of ending up in state s' by using action a in state s
                cndl_prob = self.num_tr_specific[init[0]][init[1]].get_transition_val(key) / self.num_tr_generic[init[0]][init[1]][action]
                self.tr_prob[init[0]][init[1]].set_transition_prob(key, cndl_prob)

    ## Value iteration to determine expected utilities approxly
    def __simplified_value_iteration_step__(self):
        temp_vals = (self.values).copy()
        for i in range(self.rows):
            for j in range(self.cols):
                if self.world_rep[i][j] == POS_GOAL or self.world_rep[i][j] == NEG_GOAL or self.world_rep[i][j] == NO_PATH:
                    continue
                else:
                    temp_vals[i][j] = self.reward[i][j]
                    for key in self.tr_prob[i][j].get_dictionary():
                        temp_vals[i][j] += self.DISCOUNT_FACTOR*self.tr_prob[i][j].get_transition_val(key)*self.values[key[ROW_KEY]][key[COL_KEY]]
        self.values = temp_vals

    def __check_goal_state__(self):
        if self.world_rep[self.posn[0]][self.posn[1]] == POS_GOAL or self.world_rep[self.posn[0]][self.posn[1]] == NEG_GOAL:
            self.posn = [0,0]

    def act(self):
        init_state = copy(self.posn)
        action = self.__select_move__()
        self.__update_tr__(init_state, self.posn, action)
        self.__simplified_value_iteration_step__()
        self.__check_goal_state__()
