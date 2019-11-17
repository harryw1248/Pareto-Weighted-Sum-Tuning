
import numpy as np
import helper as hp

"""
Class that simulates a sample user
User has an alpha value and tolerance value
"""
class Sample_User:

    #Notes: introduce tolerance(noise) and figure out best user_decision data structure

    alpha = 0.30
    tolerance = 0.05

    user_objective_values = {}
    user_rank_indices = {} #for ranks 0...n, n designates the best rank, 
    ordered_list = []

    def __init__(self, alpha, tolerance):
        self.alpha = alpha
        self.tolerance = tolerance
    
    def __del__(self): 
        self.user_objective_values.clear()
        self.user_rank_indices.clear() #for ranks 0...n, n designates the best rank, 
        del self.ordered_list[:]
        #print('Destructor called, Object deleted.')
    
    def get_user_objective_values(self):
        return self.user_objective_values
    
    def get_user_rank_indices(self):
        results = hp.get_rankings(self.user_objective_values, self.user_rank_indices)
        self.user_rank_indices = results[0]
        return self.user_rank_indices

    def get_user_ordered_list(self):
        results = hp.get_rankings(self.user_objective_values, self.user_rank_indices)
        self.ordered_list = results[1]
        return self.ordered_list

    #Note: how should we model fluctuation? uniform? normal?
    def user_decision(self, objective_value_pair_1):
        
        np.random.seed(101) 
        trial_alpha = np.random.uniform(self.alpha - self.tolerance, self.alpha + self.tolerance)
        multi_objective_value_1 = trial_alpha*objective_value_pair_1[0] + (1-trial_alpha)*objective_value_pair_1[1]
        self.user_objective_values[objective_value_pair_1] = multi_objective_value_1



    


    