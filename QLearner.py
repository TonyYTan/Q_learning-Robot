""""""
"""
================================================================================================================
I used the Q-Learning template from Georgia Institute of Technology. this is a public template can be found in :
https://lucylabs.gatech.edu/ml4t/spring2023/project-7/
================================================================================================================

Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""




import random as rand
import numpy as np
class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=True,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 3
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Q_table = np.zeros((num_states, num_actions))
        # self.experienced_tuple = []
        self.experienced_tuple_dict = {} # to record the experienced_tuple as a dictionary to speed up the process
        self.experienced_tuple_dict_key = 0

        if self.dyna > 0:
            # constrcut T_count & Transfer probility table
            self.t_count = np.full((num_states, num_actions, num_states), 0.00001)
            # self.t_count = np.zeros((num_states, num_actions, num_states))
            # self.t_count = self.t_count.fill(0.00001)
            # print(self.t_count)
            self.t_probility = self.t_count/np.sum(self.t_count, axis = 2, keepdims= True)
            # construct Reward table
            # self.R = self.Q_table.copy()
            # self.R = self.R.fill(0) # most of the step rewards are -1 (accroding to the project 7 document)
            self.R = np.zeros((num_states, num_actions))

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        randomly choose an action in Q table at "s"
        """
        self.s = s
        # action = rand.randint(0, self.num_actions - 1)
        action = np.argmax(self.Q_table[s])
        if self.verbose:
            print(f"s = {s}, a = {action}")
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new reward
        @returns: The selected action
        """

        # update Q-table
        # self.update_Q(self.s, self.a, s_prime, r)
        s = self.s
        a = self.a

        q_predict_value = self.Q_table[s][a]
        if np.all((self.Q_table[s_prime] == 0)):
            q_target_value = r + self.gamma * self.Q_table[s_prime][rand.randint(0, self.num_actions - 1)]
        else:
            q_target_value = r + self.gamma * self.Q_table[s_prime][np.argmax(self.Q_table[s_prime])]
        new_update_predict_value = (1 - self.alpha) * q_predict_value + self.alpha * (q_target_value)
        self.Q_table[s][a] = new_update_predict_value



        # dyna
        # if self.dyna > 0:
        #     self.execute_dyna()

        if np.random.rand(1)[0] < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q_table[s_prime])
        self.rar = self.rar  * self.radr
        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        if self.dyna > 0:
            # self.experienced_tuple.append((self.s, self.a, s_prime, r))
            current_experienced_tuple = (self.s, self.a, s_prime, r)
            self.experienced_tuple_dict[self.experienced_tuple_dict_key] = current_experienced_tuple
            self.experienced_tuple_dict_key = self.experienced_tuple_dict_key + 1
            # increment the T_count table
            self.t_count[s][a][s_prime] = self.t_count[s][a][s_prime] + 1
            # print('\n', self.t_count[s][a])
            # update the t_probility table
            # self.t_probility = self.t_count/np.sum(self.t_count, axis = 2, keepdims=True)
            self.t_probility[s][a] = self.t_count[s][a]/np.sum(self.t_count[s][a])
            # print('\n', self.t_probility[s][a])
            # update the R table
            self.R[s][a] = (1 - self.alpha) * self.R[s][a] + self.alpha * r
            random_tuple_list = np.random.randint(len(self.experienced_tuple_dict.keys()), size=self.dyna)
            # print("len(self.experienced_tuple_dict.keys()): ", len(self.experienced_tuple_dict.keys()))
            # update Q table for every hallucinate dyna cycle
            i = 0
            while i < self.dyna:
                random_tuple_current = self.experienced_tuple_dict[random_tuple_list[i]]
                s_random = random_tuple_current[0]
                a_random = random_tuple_current[1]
                s_prime_from_trans = random_tuple_current[2]
                # r_from_dyna_R_table = random_tuple_current[3]
                # s_random = rand.randint(0, self.num_states - 1)
                # a_random = rand.randint(0, self.num_actions - 1)
                # s_prime_from_trans = np.random.choice(self.num_states, 1, p = self.t_probility[s_random][a_random])[0]
                # print(i, ": current s_prine_from_T_transit is: ", s_prime_from_trans)
                r_from_dyna_R_table = self.R[s_random][a_random]
                q_predict_value_dyna = self.Q_table[s_random][a_random]
                q_target_value_dyna = r_from_dyna_R_table + self.gamma * self.Q_table[s_prime_from_trans][np.argmax(self.Q_table[s_prime_from_trans])]
                self.Q_table[s_random][a_random] = (1 - self.alpha) * q_predict_value_dyna + self.alpha * q_target_value_dyna
                i = i + 1

        self.s = s_prime
        self.a = action
        return action



    def author(self):
        return 'ytan319'


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
