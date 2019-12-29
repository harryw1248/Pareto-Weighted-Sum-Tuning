import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import helper as hp 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sample_user_rank import Sample_User
from statistics import mean 

def users_kendall_tau(user_1, user_2, objective_value_pairs):

    for example in objective_value_pairs:
        user_1.user_decision(example)
    
    for example in objective_value_pairs:
        user_2.user_decision(example)
    
    user_1_rank_indices = user_1.get_user_rank_indices()
    user_2_rank_indices = user_1.get_user_rank_indices()
    tau = hp.compute_kendall_tau(user_1_rank_indices, user_2_rank_indices)

    return tau

def user_feedback(sample_pairs):
    user_rank_indices = {}

    print("Pairs to rank")
    for item in sample_pairs:
        print(item)

    print("Give each pair a unique score from 0 to " + str(len(sample_pairs)-1))
    print("0 represents least desirable pair")
    print(str(len(sample_pairs)-1) + " represents most desirable pair\n")
    for item in sample_pairs:
        print(item)
        score = float(input())
        user_rank_indices[item] = score
    
    print("Scores given by you")
    for item in user_rank_indices:
        print(str(item) + " " + str(user_rank_indices[item]))

    coef = hp.learn_parameters(user_rank_indices, sample_pairs)
    ordered_list = [x[0] for x in hp.get_rankings(user_rank_indices, user_rank_indices)[1]]

    sum_coef = sum(coef)
    alpha_learned = coef[0]/sum_coef

    return [alpha_learned, ordered_list]

def reccomend_pairs():
    alphas_learned = []
    objective_value_pairs = hp.generate_data()

    #get 5 percent of samples from data
    data_subset = hp.get_data_subset(objective_value_pairs)
    sample_pairs = data_subset[1]

    ordered_list_user = []
    ordered_list_generated = []
    kendall_tau_scores = []

    generate = '1'
    while generate == '1' and len(objective_value_pairs) != 0:
        user_feedback_results = user_feedback(sample_pairs)
        alpha_learned = user_feedback_results[0]
        ordered_list_user = user_feedback_results[1]

        if(len(ordered_list_user) != 0 and len(ordered_list_generated) != 0):
            tau = hp.compute_kendall_tau(ordered_list_user, ordered_list_generated, list=True)
            kendall_tau_scores.append(tau)
            mean_kendall_tau = mean(kendall_tau_scores)
            print("Iteration Kendall Tau: " + str(tau))
            print("Mean Kendall Tau: " + str(mean_kendall_tau))

        objective_value_pairs = [x for x in objective_value_pairs if x not in sample_pairs]
        data_subset = hp.get_data_subset(objective_value_pairs)
        sample_pairs = data_subset[1]
        alphas_learned.append(alpha_learned)
        mean_alpha_learned = mean(alphas_learned)

        print("Current alpha value: " + str(mean_alpha_learned) + "\n")

        user_1 = Sample_User(mean_alpha_learned, 0)
        for example in sample_pairs:
            user_1.user_decision(example)

        ordered_list_generated = [x for x in user_1.get_user_ordered_list()]
        print("Reccomended ordering of newly generated pairs")
        for i in range(len(ordered_list_generated)):
            print(str(ordered_list_generated[i]) + " " + str(float(i)))
        print("Continue generating ranked pairs? Enter 1 for yes and 0 for no\n")
        del user_1
        generate = input()

    return 



