"""
Class that simulates a sample user/decision-maker
User has an alpha value that represents criteria weight and tolerance value that represents human variation
"""
import numpy as np
import pwst_util as util

class Sample_User:

    #initiliazed to default values
    alpha_vector = [0.3, 0.5]
    tolerance_vector = [0.05, 0.05] 

    #key: objective value tuple; value: score assigned to objective value tuple
    user_objective_values = {}

    #for ranks 0...n, n designates the best rank
    ordered_list = []

    def __init__(self, alpha_vector, tolerance_vector):
        self.alpha_vector = list(alpha_vector)
        self.tolerance_vector = list(tolerance_vector)
        self.user_objective_values = {}
        self.user_rank_indices = {} 
        self.ordered_list = []
    
    def __del__(self): 
        self.user_objective_values.clear()
        self.user_rank_indices.clear()
        del self.ordered_list[:]
    
    def clear_user_history(self):
        self.user_objective_values.clear()
        self.user_rank_indices.clear() 
        del self.ordered_list[:]
    
    def get_user_objective_values(self):
        """
        Returns user's criteria weights
        """
        return self.user_objective_values
    
    def get_user_ordered_list(self):
        """
        Returns objective tuples in ranked order
        """
        objective_values_list = []

        for item in self.user_objective_values:
            input = item
            output = self.user_objective_values[item]
            objective_values_list.append((input, output))
            
        objective_values_list.sort(key = lambda x: x[1])  
        ordered_list = [x for x in objective_values_list]
        self.ordered_list = [x[0] for x in ordered_list]
        return self.ordered_list

    def user_decision(self, objective_value_tuple):
        """
        Uses criteria weights to simulate decision
        """
        np.random.seed(101) 
        trial_alpha_vector = list(self.alpha_vector)
        for i in range(len(self.alpha_vector)):
            trial_alpha_vector[i] = np.random.uniform(self.alpha_vector[i] - self.tolerance_vector[i], self.alpha_vector[i] + self.tolerance_vector[i])
        trial_alpha_vector.append((1.0-sum(trial_alpha_vector)))

        #normalize trial alpha vector
        sum_trial_alpha = sum(trial_alpha_vector)
        trial_alpha_vector = [x/sum_trial_alpha for x in trial_alpha_vector]

        multi_objective_value = 0
        for i in range(len(trial_alpha_vector)):
            multi_objective_value += trial_alpha_vector[i]*objective_value_tuple[i]
        self.user_objective_values[objective_value_tuple] = multi_objective_value



    


    