import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import helper as hp 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sample_user_rank import Sample_User
import gui

trials_data = []

def users_kendall_tau(user_1, user_2, objective_value_pairs):

    for example in objective_value_pairs:
        user_1.user_decision(example)
    
    for example in objective_value_pairs:
        user_2.user_decision(example)
    
    user_1_rank_indices = user_1.get_user_rank_indices()
    user_2_rank_indices = user_1.get_user_rank_indices()
    tau = hp.compute_kendall_tau(user_1_rank_indices, user_2_rank_indices)

    return tau

def user_feedback():
    objective_value_pairs = hp.generate_data()

    #get 5 percent of samples from data
    data_subset = hp.get_data_subset(objective_value_pairs)
    objective_value_lists = data_subset[0]
    sample_pairs = data_subset[1]
    sample_pairs_list = data_subset[2]
    user_rank_indices = {}

    #gui.generate_menu(sample_pairs)
    #print("gui done")

    print("Pairs to rank")
    for item in sample_pairs:
        print(item)

    print("Give each pair a unique score from 0 to " + str(len(sample_pairs)-1))
    print("0 represents least desirable pair")
    print(str(len(sample_pairs)-1) + " represents most desirable pair")
    for item in sample_pairs:
        print(item)
        score = float(input())
        user_rank_indices[item] = score
    
    print("Scores given by you")
    for item in user_rank_indices:
        print(str(item) + " " + str(user_rank_indices[item]))

    coef = hp.learn_parameters(user_rank_indices, sample_pairs)
    print("Coefficients:")
    print(coef)
    sum_coef = sum(coef)

    print("Normalized Coefficients:")
    print([x/sum_coef for x in coef])

    return

def trial(id='Test', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=10):

    trial_data = {'alpha_true': alpha, 'tolerance': tolerance, 'func1_lower': func1_lower, 'func1_upper': func1_upper, 
                 'func2_upper': func2_upper, 'func2_lower': func2_lower, 'num_points': num_points, 'noise': noise, 
                 'weight1': 0, 'weight2': 0, 'alpha_learned': 0, 'kendall_tau': 0}

    print("Trial " + str(id))
    objective_value_pairs = hp.generate_data(id, func1_lower, func1_upper, func2_upper, func2_lower, num_points, noise)

    #get 5 percent of samples from data
    data_subset = hp.get_data_subset(objective_value_pairs)
    objective_value_lists = data_subset[0]
    sample_pairs = data_subset[1]
    sample_pairs_list = data_subset[2]

    #have virtual sample user make decisions
    user_1 = Sample_User(alpha, tolerance)

    for example in sample_pairs:
        user_1.user_decision(example)
    
    user_rank_indices = user_1.get_user_rank_indices()
    
    #use ML to analyze results
    coef = hp.learn_parameters(user_rank_indices, sample_pairs)
    print("Coefficients:")
    print(coef)
    trial_data['weight1'] = coef[0] 
    trial_data['weight2'] = coef[1] 
    sum_coef = sum(coef)

    print("Normalized Coefficients:")
    print([x/sum_coef for x in coef])
    alpha_learned = (coef[0] / sum_coef) 

    trial_data['alpha_learned'] = alpha_learned
    trial_data['alpha_error'] = ( abs(alpha_learned - alpha) / alpha ) 

    print("\n")
    trials_data.append(trial_data)

    #graph results
    for example in objective_value_pairs:
        user_1.user_decision(example)

    user_objective_values = user_1.get_user_objective_values()

    hp.graph_trial_ML_results(user_objective_values, objective_value_lists, func1_lower, func1_upper, func2_upper, func2_lower, num_points, alpha_learned, show=True)
    
    #user comparison

    learned_user = Sample_User(alpha_learned, 0)
    #tau = users_kendall_tau(user_1, learned_user, objective_value_pairs)
    #print("Kendall-Tau")
    #print(tau)
    #trial_data['kendall_tau'] = tau 

    return trial_data
    

def main():

    
    #trial(id='0.0 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=10)
    #trial(id='0.1 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=15)
    #trial(id='0.2 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=20)
    #trial(id='0.3 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=25)

    
    #trial(id='0.0 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=10)
    #trial(id='0.1 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=15)
    #trial(id='0.2 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=20)
    #trial(id='0.3 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=25)

    '''
    trial(id='1.0 50 points', alpha=0.3, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=10)
    trial(id='1.1 50 points', alpha=0.3, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=15)
    trial(id='1.2 50 points', alpha=0.3, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=20)
    trial(id='1.3 50 points', alpha=0.3, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=25)
    

    trial(id='2.0 50 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=10)
    trial(id='2.1 50 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=15)
    trial(id='2.2 50 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=20)
    trial(id='2.3 50 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=50, noise=25)

    trial(id='2.0 100 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=10)
    trial(id='2.1 100 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=15)
    trial(id='2.2 100 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=20)
    trial(id='2.3 100 points', alpha=0.5, tolerance=0.05, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=25)
    

    trial(id='3.0 100 points', alpha=0.7, tolerance=0.07, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=10)
    trial(id='3.1 100 points', alpha=0.7, tolerance=0.07, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=15)
    trial(id='3.2 100 points', alpha=0.7, tolerance=0.07, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=20)
    trial(id='3.3 100 points', alpha=0.7, tolerance=0.07, func1_lower=75, func1_upper=175, func2_upper=125, func2_lower=25, num_points=100, noise=25)
    
    trial(id='4.0 100 points', alpha=0.2, tolerance=0.03, func1_lower=5000, func1_upper=10000, func2_upper=2000, func2_lower=-1000, num_points=100, noise=10)
    trial(id='4.1 100 points', alpha=0.2, tolerance=0.03, func1_lower=5000, func1_upper=10000, func2_upper=2000, func2_lower=-1000, num_points=100, noise=15)
    trial(id='4.2 100 points', alpha=0.2, tolerance=0.03, func1_lower=5000, func1_upper=10000, func2_upper=2000, func2_lower=-1000, num_points=100, noise=20)
    trial(id='4.3 100 points', alpha=0.2, tolerance=0.03, func1_lower=5000, func1_upper=10000, func2_upper=2000, func2_lower=-1000, num_points=100, noise=25)
    '''

    df = pd.DataFrame(trials_data)
    df.to_excel("experiment_results.xlsx")
    user_feedback()


main()

