import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def compute_kendall_tau(ranking1_table, ranking2_table):
    ranking1_list = []
    ranking2_list = []

    return         

def get_rankings(objective_values, rank_indices):
    objective_values_list = []
    for item in objective_values:
        input = item
        output = objective_values[item]
        objective_values_list.append((input, output))
        
    objective_values_list.sort(key = lambda x: x[1])  
    num_items = len(objective_values)

    for pair in objective_values:
        rank_index = 0.0
        for item in objective_values_list:
            if pair != item[0]:
                rank_index += 1
            else:
                rank_indices[pair] = rank_index
    
    return rank_indices
                
def tuples_to_list(pairs):
    new_list = []
    for item in pairs:
        sub_list = [item[0], item[1]]
        new_list.append(sub_list)
    
    return new_list

def generate_data(id="Test",func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=10):
    np.random.seed(101) 
    x = np.linspace(func1_lower, func1_upper, num_points) #weighted_value
    y = np.linspace(func2_upper, func2_lower, num_points)  #worst_case
  
    x += np.random.uniform(-1*(noise), noise, num_points) 
    y += np.random.uniform(-1*(noise), noise, num_points) 

    objective_value_pairs = [] 
    counter = 0
    for item in x:
        pair = (x[counter],y[counter])
        objective_value_pairs.append(pair)
        counter += 1
    
    plt.title("Objective Value Pairs " + str(id))
    plt.xlabel('weighted_value') 
    plt.ylabel('max_worst_case') 
    plt.scatter([x[0] for x in objective_value_pairs], [y[1] for y in objective_value_pairs], color="blue") 
    #plt.show()

    return objective_value_pairs