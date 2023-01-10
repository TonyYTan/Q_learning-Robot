import numpy as np
import pandas as pd
import random as rand
import math
import time
import QLearner as QL
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=100000)

np.random.seed(200)

class Mazer(object):
    def __init__(self, verbose=False, impact=0, commission=0):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose,
        self.impact = impact,
        self.commision = commission,
        # rand.seed(2021200)
        # np.random.seed(238738)
        start_position = [0,0]
        exit_position = [0,0]
        max_moves = 100
        max_study_round = 100
        maze_heigh = 0
        maze_width = 0
        maze = 0
        reward_map = 0
        record_map = 0
        road_block_num = 0
        road_block_list = []
        stete_num = 0
        best_route = {"steps" : float('inf'),
                      "route_map": []}
        self.learner = QL.QLearner(num_states = stete_num,\
                                    num_actions = 4,  # up: 0, down: 1, left: 2, right: 3 
                                    alpha = 0.2, \
                                    gamma = 0.9, \
                                    rar = 0.95, \
                                    radr = 0.99, \
                                    dyna = 200, \
                                    # start_position = [0,0]\
                                    # exit_position = [math.sqrt(num_states)-1, math.sqrt(num_states)-1],\
                                    verbose = False)
    
    
    def set_heigh(self):
        heigh = input("Enter maze heigh:")
        print("heigh is: " + heigh)
        self.maze_heigh = int(heigh)
        return heigh
    
    def set_width(self):
        width = input("Enter maze width:")
        self.maze_width = int(width)
        print("width is: " + width)
    
    def generate_maze(self):
        # maze = np.zeros((self.maze_heigh, self.maze_width))
        maze = np.full((self.maze_heigh, self.maze_width), "_",dtype=str)
        if self.road_block_num > 0 :
            for states in self.road_block_list:
                row_num, column_num = self.decode_state(states)
                maze[row_num, column_num] = "B"
        self.maze = maze
        print(self.maze)
        return maze
    
    def set_record_map(self):
        record_map = np.full((self.maze_heigh, self.maze_width), "_",dtype=str)
        for states in self.road_block_list:
            row_num, column_num = self.decode_state(states)
            record_map[row_num, column_num] = "B"
        self.record_map = record_map
        return
    
    def update_max_moves(self):
        new_max_moves = (self.maze_heigh + self.maze_width) * 5
        self.max_moves = new_max_moves
        return 
    
    def get_heigh(self):
        return self.maze_heigh
    
    def get_width(self):
        return self.maze_width
    
    def update_exit_position(self):
        self.exit_position = [self.maze_heigh - 1, self.maze_width - 1]
        # print(self.exit_position)
        # self.maze[self.exit_position[0], self.exit_position[1]] = 1
        # print(self.maze)
        return self.exit_position
    
    def reset_initial_poisition_and_random_action(self):
        self.learner.s = 0
        self.learner.a = np.random.randint(0, 4)
        # self.learner.alpha = 0.2
        # self.learner.gamma = 0.9
        # self.learner.rar = 0.95
        # self.learner.radr = 0.99
        return
    
    def set_max_study_round(self):
        max_study_round = input("How many study round can Q-Learning robot train: ")
        print("max study round is : " + max_study_round)
        self.max_study_round = int(max_study_round)
        return self.max_study_round 
    
    def define_num_of_state(self):
        self.state_num = (self.exit_position[0]) * (self.exit_position[1] + 1) + (self.exit_position[1] + 1)
        self.learner.num_states = self.state_num
        self.learner = QL.QLearner(num_states = self.state_num,\
                            num_actions = 4,  # up: 0, down: 1, left: 2, right: 3 
                            alpha = 0.2, \
                            gamma = 0.9, \
                            rar = 0.95, \
                            radr = 0.99, \
                            dyna = 200, \
                            # start_position = [0,0]\
                            # exit_position = [math.sqrt(num_states)-1, math.sqrt(num_states)-1],\
                            verbose = False)
        # print("Q-table: ", '\n', self.learner.Q_table)
    
    def code_state(self, position):
        state_num = position[0] * (self.maze_width) + (position[1])
        return state_num
    
    def decode_state(self, state):
        row_num = state // self.maze_width
        column_num = state % self.maze_width
        return (row_num, column_num)
    
    def reward_map(self):
        reward_map = np.full((self.maze_heigh, self.maze_width), -1)
        reward_map[self.maze_heigh -1][self.maze_width - 1] = 1000
        self.reward_map = reward_map
        # print(self.reward_map)
        return 
    
    def set_random_road_block_num(self):
        road_block_num = input("How many random road block: (from 0 to {} )".format((self.maze_heigh * self.maze_width)/3))
        # print("random road block num is: " + road_block_num)
        road_block_num = int(road_block_num)
        if road_block_num < 0:
            road_block_num = 0
        elif road_block_num >= ((self.maze_heigh * self.maze_width)/3):
            road_block_num = ((self.maze_heigh * self.maze_width)/3)//1
        self.road_block_num = int(road_block_num)
        print("random road block num is: ", road_block_num)
        return self.road_block_num

    def generate_random_road_block_list(self):
        exit_position = self.exit_position
        exit_position_state = self.code_state(exit_position)
        road_block_num = self.road_block_num
        road_block_list = []
        for i in range (0, road_block_num):
            random_state = np.random.randint(1, (exit_position_state - 1))
            road_block_list.append(random_state)
        self.road_block_list = road_block_list
        return road_block_list
    
    def set_gold_position(self):
        gold_position_row = input(('Please set gold position in row No: must be from 0 to ',  self.maze_heigh - 1))
        gold_position_row = int(gold_position_row)
        if gold_position_row < 0:
            gold_position_row =0
        elif gold_position_row > (self.maze_heigh -1) :
            gold_position_row = self.maze_heigh - 1
            
        gold_position_col = input(("Please set gold position in column No: (must be from 0 to ", self.maze_width - 1, ")"))
        gold_position_col = int(gold_position_col)
        if gold_position_col < 0:
                gold_position_col =0
        elif gold_position_col > (self.maze_width -1) :
            gold_position_col = self.maze_width - 1
        self.reward_map[gold_position_row][gold_position_col] = 50
        return 
    
    def get_reward(self, state):
        row_num, column_num = self.decode_state(state)
        reward = self.reward_map[row_num, column_num]
        return reward

# about the road block
    def state_under_road_block(self):
        target_list = []
        for states in self.road_block_list: 
            row_num, column_num = self.decode_state(states)
            target_row_num = row_num + 1
            target_column_num = column_num 
            target_state = self.code_state([target_row_num, target_column_num])
            target_list.append(target_state)
        return target_list
    
    def state_above_road_block(self):
        target_list = []
        for states in self.road_block_list: 
            row_num, column_num = self.decode_state(states)
            target_row_num = row_num - 1
            target_column_num = column_num 
            target_state = self.code_state([target_row_num, target_column_num])
            target_list.append(target_state)
        return target_list
    
    def state_on_road_block_left(self):
        target_list = []
        for states in self.road_block_list: 
            row_num, column_num = self.decode_state(states)
            target_row_num = row_num 
            target_column_num = column_num - 1
            target_state = self.code_state([target_row_num, target_column_num])
            target_list.append(target_state)
        return target_list
    
    def state_on_road_block_right(self):
        target_list = []
        for states in self.road_block_list: 
            row_num, column_num = self.decode_state(states)
            target_row_num = row_num 
            target_column_num = column_num + 1
            target_state = self.code_state([target_row_num, target_column_num])
            target_list.append(target_state)
        return target_list
        
    def initialize_best_route_dict(self):
        self.best_route = {"steps" : float('inf'),
                            "route_map": []}
        return
    
    def update_best_route(self, step_count):
        if step_count < self.best_route["steps"]:
            self.best_route["steps"] = step_count
            self.best_route["route_map"] = self.record_map
        return
    
# env reaction
    def is_terminated(self, state):
        if state == (np.shape(self.learner.Q_table)[0] -1):
            return True
        else:
            return False
        
    def get_env_feedback(self, state, action):
        # heigh = self.maze_heigh
        # width = self.maze_width
        row_num = state // self.maze_width
        column_num = state % self.maze_width
        # in top row (or under the road block)
        if row_num == 0 or state in self.state_under_road_block():
            # can not move up
            if action == 0:
                state_prime = state
                return state_prime
        # in bottom row
        if row_num == np.shape(self.maze)[0] - 1 or state in self.state_above_road_block():
            # can not move down
            if action == 1:
                state_prime = state
                return state_prime
        # in left column
        if column_num == 0 or state in self.state_on_road_block_right():
            # can not move left
            if action == 2:
                state_prime = state
                return state_prime
        # in right column
        if column_num == np.shape(self.maze)[1] - 1 or state in self.state_on_road_block_left():
            # can not move right
            if action == 3:
                state_prime = state
                return state_prime
        # else can move up/down/left/right
        if action == 0: #move up
            state = (row_num - 1) * np.shape(self.maze)[1] + column_num
            return state
        elif action == 1: #move down
            state = (row_num + 1) * np.shape(self.maze)[1] + column_num
            return state
        elif action == 2: #move left
            state = (row_num) * np.shape(self.maze)[1] + (column_num - 1)
            return state
        elif action == 3: #move right
            state = (row_num) * np.shape(self.maze)[1] + (column_num + 1)
            return state

# presenting on screen
    def np_df_to_present(self):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        LINE_DOWN = '\033[1B'
        # maze_to_present = self.maze
        maze_to_present = np.full((self.maze_heigh, self.maze_width), " ",dtype=str)
        for states in self.road_block_list:
            row_num, column_num = self.decode_state(states)
            maze_to_present[row_num, column_num] = "B"
        state_to_present = self.learner.s
        position = self.decode_state(state_to_present)
        maze_to_present[position[0], position[1]] = "*"
        # print(maze, end= '\r')
        self.record_map[position[0], position[1]] = "*"
        print(np.array2string(maze_to_present, separator=' ', formatter={'str_kind': lambda x: x}), end = "\r" + LINE_UP * (self.maze_heigh - 1))
        # print(self.maze)
        # sys.stdout.write("\033[K")
        # print(position)
        time.sleep(0.05)

    def testing_random(self, state):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        LINE_DOWN = '\033[1B'
        move_count = 0
        self.set_record_map()
        while self.is_terminated(state) == False and move_count < self.max_moves:
            # print("not ending yet")
            self.learner.s = state
            self.np_df_to_present()
            action = np.random.randint(0, 4)
            # print("action: ", action)
            state = self.get_env_feedback(state, action)
            move_count += 1
        if self.is_terminated(state) == True:
            print(LINE_DOWN * (self.maze_heigh) + "======================Congraduations!===================")
            print("for random movement teating: Q Learning robot used ", move_count, " to find the exit poisition!!")
            self.record_map[self.maze_heigh - 1][self.maze_width - 1] = "*"
            print(np.array2string(self.record_map, separator=' ', formatter={'str_kind': lambda x: x}))
        else:
            # print(LINE_DOWN * (self.maze_heigh - 1) + "reach maxing moving, but still not getting the exit position")
            print("reach maxing moving, but still not getting the exit position")
        return
    
    def q_learning_robot(self):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        LINE_DOWN = '\033[1B'
        move_count = 0
        state = self.learner.s
        action = self.learner.a
        self.set_record_map()
        while self.is_terminated(self.learner.s) == False and move_count < self.max_moves:
            self.np_df_to_present()
            r = self.get_reward(self.learner.s)
            s_prime = self.get_env_feedback(self.learner.s, self.learner.a)
            action = self.learner.query(s_prime, r)
            move_count += 1
        if self.is_terminated(self.learner.s) == True:
            print(LINE_DOWN * (self.maze_heigh - 1) + "======================Congraduations!===================")
            print("for random movement testing: Q Learning robot used ", move_count, " to find the exit poisition!!")
            self.record_map[self.maze_heigh - 1][self.maze_width - 1] = "*"
            print(np.array2string(self.record_map, separator=' ', formatter={'str_kind': lambda x: x}))
            print('\n', '\n')
            self.update_best_route(move_count)
        else:
            print(LINE_DOWN * (self.maze_heigh - 1) + "reach maxing moving, but still not getting the exit position")
        return



    def testing(self):
        while self.is_terminated(self.learner.s) == False:
            self.np_df_to_present()
            self.learner.s += 1
            # time.sleep(0.5)
            # for i in range(10):
            # if i % 2 == 0:
            # print(i, '\n', i, end="\r")
            # time.sleep(1) 

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



if __name__ == "__main__":
    print("Maze Runner!!!")
    import Mazer as mz
    learner = mz.Mazer()
    learner.set_heigh()
    learner.set_width()
    learner.update_max_moves()
    learner.update_exit_position()
    learner.define_num_of_state()
    learner.set_random_road_block_num()
    learner.set_max_study_round()
    learner.generate_random_road_block_list()
    learner.generate_maze()
    learner.initialize_best_route_dict()
    # learner.np_df_to_present()
    # learner.testing()
    learner.reward_map()
    # learner.set_gold_position()
    # print(learner.reward_map)
    # learner.testing_random(0)
    counter = 1
    while counter < learner.max_study_round:
        print("Round ", counter)
        learner.q_learning_robot()
        counter += 1
        # print(learner.learner.Q_table)
        learner.reset_initial_poisition_and_random_action()
    print('\n', "========================== Final Summary ==========================")
    print("the best route used ", learner.best_route["steps"], " and the route map is: ", '\n')
    print(np.array2string(learner.best_route["route_map"], separator=' ', formatter={'str_kind': lambda x: x}))
    print()
