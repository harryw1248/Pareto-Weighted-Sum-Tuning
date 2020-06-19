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
    #num_items = len(objective_values)

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

def dominates(pair1, pair2):
    if pair1[0] > pair2[0] and pair1[1] > pair2[1]:
        return True
    return False

def get_non_dominated(objective_value_pairs):
    non_dominated_pairs = []
    for i in range(len(objective_value_pairs)):
        dominated = False
        for j in range(len(objective_value_pairs)):
            if dominates(objective_value_pairs[j], objective_value_pairs[i]):
                dominated = True
        if dominated == False:
            non_dominated_pairs.append(objective_value_pairs[i])
    non_dominated_pairs = sorted(non_dominated_pairs, key=lambda x: x[0])
    print("Number of Non-Dominated Pairs: " + str(len(non_dominated_pairs)))
    return non_dominated_pairs

'''
number of data-points sampled at each iteration equals 2*margin_from_half + 1
'''
def get_data_subset(objective_value_tuples, margin_from_half, random_sampling):
    np.random.seed(101) 
    sample_tuples = []
    sample_tuples_list = []
    num_data_points = len(objective_value_tuples)
    half_point = int(num_data_points / 2)
    objective_value_lists = tuples_to_list(objective_value_tuples)

    if random_sampling == False:
        for i in range(half_point-margin_from_half,half_point+margin_from_half+1):
            sample_tuples.append(objective_value_tuples[i])
            sample_tuples_list.append(objective_value_lists[i])
    
    

    if random_sampling == True:
        for i in range(margin_from_half*2+1):
            sample_tuples.append(random.choice(objective_value_tuples))
    
    
    #sample_pairs_list = tuples_to_list(sample_pairs)
    
    return [objective_value_lists, sample_tuples, sample_tuples_list]

def create_latex_files():
    f_random = open("random_sampling.txt", "w")
    f_random.write("Data-points Sampled at Random\\\\" + "\n")
    f_random.write("\\begin{tabular}{ccccc}" + "\n")
    f_random.write("\\hline" + "\n")
    f_random.write("$x$ & $RE_{1}$ & $RE_{5}$ & $RE_{10}$ & $RE_{15}$\\\\" + "\n")
    f_random.write("\\hline" + "\n")
    f_random.close()

    f_middle = open("middle_sampling.txt", "w")
    f_middle.write("Data-points Sampled from Middle First\\\\" + "\n")
    f_middle.write("\\begin{tabular}{ccccc}" + "\n")
    f_middle.write("\\hline" + "\n")
    f_middle.write("$x$ & $RE_{1}$ & $RE_{5}$ & $RE_{10}$ & $RE_{15}$\\\\" + "\n")
    f_middle.write("\\hline" + "\n")
    f_middle.close()

    return

def populate_latex_files(df):    
    f_random = open("random_sampling.txt", "a")
    f_middle = open("middle_sampling.txt", "a")
    
    for index, row in df.iterrows():
        if row['random_sampling'] == True:
            f_random.write(str(row['num_points_each_iteration']) + " & " + 
                    str(format(row['relative_alpha_error_after_iteration_1']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_5']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_10']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_15']*100,'.4g')) + "\\\\" + "\n")
        else:
            f_middle.write(str(row['num_points_each_iteration']) + " & " + 
                    str(format(row['relative_alpha_error_after_iteration_1']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_5']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_10']*100,'.4g')) + " & " +
                    str(format(row['relative_alpha_error_after_iteration_15']*100,'.4g')) + "\\\\" + "\n")

    
    f_random.close()
    f_middle.close()

    return

def finish_latex_files():
    f_random = open("random_sampling.txt", "a")
    f_random.write("\\hline" + "\n")
    f_random.write("\\end{tabular}")
    f_random.close()

    f_middle = open("middle_sampling.txt", "a")
    f_middle.write("\\hline" + "\n")
    f_middle.write("\\end{tabular}")
    f_middle.close()

    return