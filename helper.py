import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#import rpy2

def learn_parameters(user_rank_indices, sample_pairs):
    user_rank_indices_list = []
    for pair in sample_pairs:
        user_rank_indices_list.append(user_rank_indices[pair])
    
    scaler = StandardScaler()
    scaler.fit(sample_pairs)
    for pair in sample_pairs:
        scaler.transform(pair)

    reg = LinearRegression().fit(sample_pairs, user_rank_indices_list)
    score = reg.score(sample_pairs, user_rank_indices_list)
    coef = reg.coef_
    return coef

def compute_kendall_tau(ranking_1_table, ranking_2_table):
    ranking_1_list = [x[0] for x in sorted(ranking_1_table.items(), key = lambda kv:(kv[1], kv[0]))]
    ranking_2_list = [x[0] for x in sorted(ranking_2_table.items(), key = lambda kv:(kv[1], kv[0]))]
    tau = 0 #rpy2.cor(ranking_1_list, ranking_2_list, method="kendall")

    return tau         

#note: for ranking index, a higher index suggests a better ranking
#uses multi-objective value to rank pairs
def get_rankings(objective_values, rank_indices):
    objective_values_list = []
    for item in objective_values:
        input = item
        output = objective_values[item]
        objective_values_list.append((input, output))
        
    objective_values_list.sort(key = lambda x: x[1])  
    #num_items = len(objective_values)

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

def get_data_subset(objective_value_pairs):
    sample_pairs = []
    sample_pairs_list = []
    num_data_points = len(objective_value_pairs)
    half_point = int(num_data_points / 2)
    margin_from_half = int(num_data_points / 20)
    objective_value_lists = tuples_to_list(objective_value_pairs)

    for i in range(half_point-margin_from_half,half_point+margin_from_half+1):
        sample_pairs.append(objective_value_pairs[i])
        sample_pairs_list.append(objective_value_lists[i])

    return [objective_value_lists, sample_pairs, sample_pairs_list]

def graph_trial_ML_results(user_objective_values, objective_value_lists, func1_lower, func1_upper, func2_upper, func2_lower, num_points, alpha_learned, show=False):
    x_values = [x[0] for x in objective_value_lists]
    y_values = [y[1] for y in objective_value_lists]
    z_values = []

    for i in range(len(x_values)):
        key = (x_values[i], y_values[i])
        value = user_objective_values[key]
        z_values.append(value)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x_values, y_values, z_values, cmap='Greens')
    ax.set_xlabel('Weighted Value')
    ax.set_ylabel('Max Worst Case')
    ax.set_zlabel('Multi-Objective Value')

    def f(alpha, x, y):
        return alpha*x + (1-alpha)*y
    
    x = np.linspace(func1_lower, func1_upper, num_points)
    y = np.linspace(func2_upper, func2_lower, num_points)
    X, Y = np.meshgrid(x, y)
    Z = f(alpha_learned, X, Y)

    ax.contour3D(X, Y, Z, 50, cmap='binary')

    if show == True:
        plt.show()

    return