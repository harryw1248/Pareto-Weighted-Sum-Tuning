import os
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
        

#note: for ranking index, a higher index suggests a better ranking
#uses multi-objective value to rank pairs
def get_rankings(objective_values, rank_indices):
    objective_values_list = []
    for item in objective_values:
        input = item
        output = objective_values[item]
        objective_values_list.append((input, output))
        
    objective_values_list.sort(key = lambda x: x[1])  
    ordered_list = [x for x in objective_values_list]

    for pair in objective_values:
        rank_index = 0.0
        for item in objective_values_list:
            if pair != item[0]:
                rank_index += 1
            else:
                rank_indices[pair] = rank_index
    
    return (rank_indices, ordered_list)
                
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
    np.random.seed(101) 
    sample_tuples = []
    sample_tuples_list = []
    objective_value_lists = tuples_to_list(objective_value_tuples)

    for i in range(points_per_iteration):
        sample_tuples.append(random.choice(objective_value_tuples))
    
    return [objective_value_lists, sample_tuples, sample_tuples_list]

def user_feedback(sample_pairs, user_virtual, iteration_number):
    user_rank_indices = {}

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
        user_rank_indices[(ordered_list_generated[i])] = float(i)
    
    f.close()

    os.system("./svm_rank/svm_rank_learn -c 3 svm_rank/user_queries_train.dat svm_rank/model > /dev/null 2>&1")

    f_read = open("svm_rank/model", "r")
    last_line = f_read.readlines()[-1]
    arr = last_line.split(' #')[0].split()
    arr = arr[1:]
    new_arr = []
    ''' Extract each feature from the feature vector '''
    for el in arr:
        new_arr.append(float(el.split(':')[1]))
    f_read.close()

    ordered_list = [x[0] for x in get_rankings(user_rank_indices, user_rank_indices)[1]]

    sum_coef = sum(new_arr)
    alpha_vector_learned = [x/sum_coef for x in new_arr]
    alpha_vector_learned.pop()

    return [alpha_vector_learned, ordered_list]