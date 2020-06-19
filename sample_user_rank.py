
import numpy as np
import helper as hp

"""
Class that simulates a sample user
User has an alpha value and tolerance value
"""
class Sample_User:

    #Notes: introduce tolerance(noise) and figure out best user_decision data structure

    alpha_vector = [0.3, 0.5]
    tolerance_vector = [0.05, 0.05] 


    user_objective_values = {}
    user_rank_indices = {} #for ranks 0...n, n designates the best rank, 
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
        return self.user_objective_values
    
    def get_user_rank_indices(self):
        results = hp.get_rankings(self.user_objective_values, self.user_rank_indices)
        self.user_rank_indices = results[0]
        return self.user_rank_indices

    def get_user_ordered_list(self):
        results = hp.get_rankings(self.user_objective_values, self.user_rank_indices)
        self.ordered_list = [x[0] for x in results[1]]
        return self.ordered_list

    def user_decision(self, objective_value_tuple):
        
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



    


    