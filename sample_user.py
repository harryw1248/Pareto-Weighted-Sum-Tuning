
import numpy as np

"""
Class that simulates a sample user
User has an alpha value and tolerance value
"""
class Sample_User:

    #Notes: introduce tolerance(noise) and figure out best user_decision data structure

    alpha = 0.30
    tolerance = 0.05

    user_decisions = {}

    def __init__(self, alpha, tolerance):
        self.alpha = alpha
        self.tolerance = tolerance
    
    def get_user_decisions(self):
        return self.user_decisions

    """
    Simulates a user making a decision
    objective_value_pair_1 is a pair of floats
    objective_value_pair_2 is a pair of floats
    """
    #Note: how should we model fluctuation? uniform? normal?
    def user_decision(self, objective_value_pair_1, objective_value_pair_2):
        
        np.random.seed(101) 
        trial_alpha = np.random.uniform(self.alpha - self.tolerance, self.alpha + self.tolerance)
        multi_objective_value_1 = trial_alpha*objective_value_pair_1[0] + (1-trial_alpha)*objective_value_pair_1[1]
        multi_objective_value_2 = trial_alpha*objective_value_pair_2[0] + (1-trial_alpha)*objective_value_pair_2[1]

        if multi_objective_value_1 >= multi_objective_value_2:
            self.user_decisions[objective_value_pair_1] = 1
            self.user_decisions[objective_value_pair_2] = -1
            return
        
        self.user_decisions[objective_value_pair_1] = -1
        self.user_decisions[objective_value_pair_2] = 1

    


    