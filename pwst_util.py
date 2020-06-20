"""
Contains utility functions for intermediate computations in Pareto-Weighted-Sum-Tuning
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
                
def tuples_to_list(pairs):
    new_list = []
    for item in pairs:
        sub_list = [item[0], item[1]]
        new_list.append(sub_list)
    
    return new_list

def average_vectors(vectors):
    result = []
    for i in range(len(vectors[0])):
        component_sum = 0
        for vector in vectors:
            component_sum += vector[i]
        result.append((component_sum/len(vectors)))

    return result

def get_data_subset(objective_value_tuples, points_per_iteration):
    """
    Samples a subset of a data-set to give to user at one iteration of PWST
    """
    np.random.seed(101) 
    sample_tuples = []
    sample_tuples_list = []
    objective_value_lists = tuples_to_list(objective_value_tuples)

    for i in range(points_per_iteration):
        sample_tuples.append(random.choice(objective_value_tuples))
    
    return [objective_value_lists, sample_tuples, sample_tuples_list]

def user_feedback(sample_pairs, user_virtual, iteration_number):
    """
    Uses Ranking-SVM to return criteria weights at one iteration
    """
    for example in sample_pairs:
        user_virtual.user_decision(example)

    ordered_list_generated = [x for x in user_virtual.get_user_ordered_list()]

    f = open("svm_rank/user_queries_train.dat", "a")
    f.write("# query " + str(iteration_number) + "\n")
    for i in range(len(ordered_list_generated)):
        f.write(str(i) + " qid:" + str(iteration_number) + " ")
        for j in range(len(ordered_list_generated[i])):
            f.write(str(j+1) + ":" + str(ordered_list_generated[i][j]) + " ")
        f.write("\n")
    
    f.close()

    os.system("./svm_rank/svm_rank_learn -c 3 svm_rank/user_queries_train.dat svm_rank/model > /dev/null 2>&1")

    f_read = open("svm_rank/model", "r")
    last_line = f_read.readlines()[-1]
    arr = last_line.split(' #')[0].split()
    arr = arr[1:]
    new_arr = []

    #Extract each feature from the feature vector
    for el in arr:
        new_arr.append(float(el.split(':')[1]))
    f_read.close()

    sum_coef = sum(new_arr)
    alpha_vector_learned = [x/sum_coef for x in new_arr]
    alpha_vector_learned.pop()

    return alpha_vector_learned
