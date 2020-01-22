import os
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

def user_feedback(sample_tuples, iteration_number):
    user_rank_indices = {}

    print("Tuples to rank")
    for item in sample_tuples:
        print(item)

    print("Give each tuple a unique score from 0 to " + str(len(sample_tuples)-1))
    print("0 represents least desirable tuple")
    print(str(len(sample_tuples)-1) + " represents most desirable tuple\n")
    for item in sample_tuples:
        print(item)
        score = float(input())
        user_rank_indices[item] = score
    
    print("Scores given by you")
    for item in user_rank_indices:
        print(str(item) + " " + str(user_rank_indices[item]))
    
    ordered_list_generated = [(item,user_rank_indices[item]) for item in user_rank_indices]
    ordered_list_generated = sorted(ordered_list_generated, key=lambda x: x[1])
    ordered_list_generated = [x[0] for x in ordered_list_generated]

    ###
    f = open("user_queries_train.dat", "a")
    f.write("# query " + str(iteration_number) + "\n")
    for i in range(len(ordered_list_generated)):
        print(str(ordered_list_generated[i]) + " " + str(float(i)))
        f.write(str(i) + " qid:" + str(iteration_number) + " ")
        for j in range(len(ordered_list_generated[i])):
            f.write(str(j+1) + ":" + str(ordered_list_generated[i][j]) + " ")
        f.write("\n")
        user_rank_indices[(ordered_list_generated[i])] = float(i)
    
    f.close()

    os.system("./svm_rank_learn -c 3 user_queries_train.dat model > /dev/null 2>&1")

    f_read = open("model", "r")
    last_line = f_read.readlines()[-1]
    arr = last_line.split(' #')[0].split()
    arr = arr[1:]
    new_arr = []
    ''' Extract each feature from the feature vector '''
    for el in arr:
        new_arr.append(float(el.split(':')[1]))
    f_read.close()

    ordered_list = [x[0] for x in hp.get_rankings(user_rank_indices, user_rank_indices)[1]]

    sum_coef = sum(new_arr)
    alpha_vector_learned = [x/sum_coef for x in new_arr]
    alpha_vector_learned.pop()

    return [alpha_vector_learned, ordered_list]

    ###

def reccomend_pairs(alpha_vector=[0.2, 0.5], tolerance_vector=[0.05, 0.05]):
    f = open("user_queries_train.dat", "w")
    f.close()

    user_virtual = Sample_User(alpha_vector, tolerance_vector)

    alpha_vectors_learned = []
    mean_alpha_vectors = [] #used to track progress of reccomendation system

    objective_value_tuples = hp.generate_data(range_vector = [100, 150, 75, -50, 75, -50], num_points=150, noise=10)
    #objective_value_tuples = hp.generate_data(range_vector = [100, 200, 80, -50, 80, -20], num_points=150, noise=10)

    #get 10 samples from data
    data_subset = hp.get_data_subset(objective_value_tuples)
    sample_tuples = data_subset[1]

    ordered_list_user = []
    ordered_list_generated = []
    kendall_tau_scores = []
    iteration_number = 0
    iteration_limit = 7
    generate = '1' #used for interative feedback
    while iteration_number < iteration_limit and len(objective_value_tuples) != 0:
        print("iteration number: " + str(iteration_number))
        iteration_number += 1
        user_feedback_results = user_feedback(sample_tuples, iteration_number)
        alpha_vector_learned = user_feedback_results[0]

        for i in range(len(alpha_vector_learned)):
            print("Iteration alpha " + str(i) + " learned: " + str(alpha_vector_learned[i]))

        ordered_list_user = user_feedback_results[1]

        objective_value_pairs = [x for x in objective_value_tuples if x not in sample_tuples]
        data_subset = hp.get_data_subset(objective_value_pairs)
        sample_tuples = data_subset[1]
        alpha_vectors_learned.append(alpha_vector_learned)
        mean_alpha_vector_learned = hp.average_vectors(alpha_vectors_learned)

        mean_alpha_vectors.append(mean_alpha_vector_learned)
        for i in range(len(mean_alpha_vector_learned)):
            print("Current mean alpha " + str(i) + " learned: " + str(mean_alpha_vector_learned[i]))

        user_1 = Sample_User(mean_alpha_vector_learned, [0 for i in range(len(mean_alpha_vector_learned))])
        for example in sample_tuples:
            user_1.user_decision(example) #need ordered list and ranks after this?

        ordered_list_generated = [x for x in user_1.get_user_ordered_list()]

        print("Reccomended ordering of newly generated pairs")
        for i in range(len(ordered_list_generated)):
            print(str(ordered_list_generated[i]) + " " + str(float(i)))
        print("\n")

        print("Continue generating ranked pairs? Enter 1 for yes and 0 for no\n")
        del user_1
        user_virtual.clear_user_history()
        generate = input()

    return 




