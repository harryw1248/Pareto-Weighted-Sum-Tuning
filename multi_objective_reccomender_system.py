import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import helper as hp 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sample_user_rank import Sample_User
from statistics import mean 
import interactive as it


trials_data = []

def users_kendall_tau(user_1, user_2, objective_value_pairs):

    for example in objective_value_pairs:
        user_1.user_decision(example)
    
    for example in objective_value_pairs:
        user_2.user_decision(example)
    
    user_1_rank_indices = user_1.get_user_rank_indices()
    user_2_rank_indices = user_2.get_user_rank_indices()
    tau = hp.compute_kendall_tau(user_1_rank_indices, user_2_rank_indices)

    return tau

def user_feedback(sample_pairs, user_virtual, iteration_number):
    user_rank_indices = {}

    
    for example in sample_pairs:
        user_virtual.user_decision(example)

    ordered_list_generated = [x for x in user_virtual.get_user_ordered_list()]

    print("Virtual User Decisions")
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

def reccomend_pairs(alpha_vector=[0.3], tolerance_vector=[0.01]):
    f = open("user_queries_train.dat", "w")
    f.close()

    user_virtual = Sample_User(alpha_vector, tolerance_vector)

    alpha_vectors_learned = []
    mean_alpha_vectors = [] #used to track progress of reccomendation system
    #objective_value_pairs = hp.generate_data(func1_lower=200, func1_upper=300, func2_upper=150, func2_lower=-50, num_points=500, noise=10)
    #objective_value_pairs = hp.generate_data(func1_lower=200, func1_upper=400, func2_upper=150, func2_lower=-50, num_points=500, noise=20)

    #objective_value_pairs = hp.generate_data(func1_lower=100, func1_upper=150, func2_upper=75, func2_lower=50, num_points=50, noise=5) #best performance so far < 0.01 error (150 points)
    #objective_value_pairs = hp.generate_data(func1_lower=200, func1_upper=400, func2_upper=200, func2_lower=-50, num_points=100, noise=20)
    #objective_value_pairs = hp.generate_data(func1_lower=200, func1_upper=400, func2_upper=200, func2_lower=-50, num_points=1000, noise=20)
    
    #objective_value_pairs = hp.generate_data(func1_lower=200, func1_upper=400, func2_upper=200, func2_lower=-50, num_points=100, noise=20)
    
    #objective_value_tuples = hp.generate_data(range_vector = [100, 150, 75, -50, 75, -50], num_points=150, noise=10)
    #objective_value_tuples = hp.generate_data(range_vector = [100, 200, 80, -50, 80, -20], num_points=150, noise=10)
    objective_value_tuples = hp.generate_data(range_vector = [100, 150, 75, -50], num_points=500, noise=20)
    objective_value_tuples = hp.generate_data(range_vector = [100, 200, 100, -50], num_points=500, noise=20) #figures 1-4 with 11 and 7 points

    #get 10 samples from data
    data_subset = hp.get_data_subset(objective_value_tuples)
    sample_tuples = data_subset[1]

    ordered_list_user = []
    ordered_list_generated = []
    kendall_tau_scores = []
    iteration_number = 0
    iteration_limit = 10
    generate = '1' #used for interative feedback
    while iteration_number < iteration_limit and len(objective_value_tuples) != 0:
        print("iteration number: " + str(iteration_number))
        iteration_number += 1
        user_feedback_results = user_feedback(sample_tuples, user_virtual, iteration_number)
        alpha_vector_learned = user_feedback_results[0]

        for i in range(len(alpha_vector_learned)):
            print("Iteration alpha " + str(i) + " learned: " + str(alpha_vector_learned[i]))

        ordered_list_user = user_feedback_results[1]

        if(len(ordered_list_user) != 0 and len(ordered_list_generated) != 0):
            tau = 0.5#hp.compute_kendall_tau(ordered_list_user, ordered_list_generated, list=True)
            kendall_tau_scores.append(tau)
            mean_kendall_tau = mean(kendall_tau_scores)
            #print("Iteration Kendall Tau: " + str(tau))
            #print("Mean Kendall Tau: " + str(mean_kendall_tau))

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

        #print("Continue generating ranked pairs? Enter 1 for yes and 0 for no\n")
        del user_1
        user_virtual.clear_user_history()
        #generate = input()

    for i in range(len(mean_alpha_vectors[0])):
        y_quantities_1 = [mean_alpha_vector[i] for mean_alpha_vector in mean_alpha_vectors]
        title = "Alpha " + str(i) + " Progress"
        plt.title(title)
        plt.xlabel("Iteration Number") 
        plt.ylabel("Alpha")
        true_alpha = plt.axhline(y=alpha_vector[i], color='r', linestyle='-', label="True Alpha Value")
        alpha_learned = plt.plot([i for i in range(iteration_limit)], y_quantities_1, label="Alpha Learned")
        plt.legend()
        plt.show()

        
        title = "Alpha " + str(i) + " Error Progress"
        y_quantities_2 = [(abs(x-alpha_vector[i])/alpha_vector[i])*100 for x in y_quantities_1]
        plt.title(title)
        plt.xlabel("Iteration Number") 
        plt.ylabel("Alpha Error Percentage")
        plt.plot([i for i in range(iteration_limit)], y_quantities_2)
        plt.show()
        


    return 

    

def main():

    #reccomend_pairs()
    if len(sys.argv) == 2 and sys.argv[1] == "it":
        it.reccomend_pairs()
    else:
        reccomend_pairs()


main()

