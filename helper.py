import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#import rpy2

def learn_parameters(user_rank_indices, sample_pairs):
    sample_pairs_list = tuples_to_list(sample_pairs)
    user_rank_indices_list = []
    for pair in sample_pairs:
        user_rank_indices_list.append(user_rank_indices[pair])
    
    #scaler = StandardScaler()
    #scaler.fit(sample_pairs_list)

    #sample_pairs_list = scaler.transform(sample_pairs_list)
    #for pair in sample_pairs_list:
    #    scaler.transform(pair)

    #for linear regression, try looking into non-negative weights as an optimization constraint later
    reg = LinearRegression().fit(sample_pairs_list, user_rank_indices_list)
    score = reg.score(sample_pairs, user_rank_indices_list)
    coef = reg.coef_
    return coef

def compute_kendall_tau(ranking_1_table, ranking_2_table, list=False):

    vector_one = ranking_1_table
    vector_two = ranking_2_table

    if list == False:
        vector_one = [x[0] for x in sorted(ranking_1_table.items(), key = lambda kv:(kv[1], kv[0]))]
        vector_two = [x[0] for x in sorted(ranking_2_table.items(), key = lambda kv:(kv[1], kv[0]))]
    #tau = 0 #rpy2.cor(ranking_1_list, ranking_2_list, method="kendall")
    
    vector_one = [str(x) for x in vector_one]
    vector_two = [str(x) for x in vector_two]

    # Generate all possible pairs of two ranks for vector_one and append them to a list
    pairs_vec_one = []
    for iter in range(len(vector_one)):
        for iter2 in range(iter + 1, len(vector_one)):
            tup = (vector_one[iter],) + (vector_one[iter2],)
            pairs_vec_one.append(tup)

    # Generate all possible pairs of two ranks for vector_two and append them to a list
    pairs_vec_two = []
    for iter in range(len(vector_two)):
        for iter2 in range(iter + 1, len(vector_two)):
            tup = (vector_two[iter],) + (vector_two[iter2],)
            pairs_vec_two.append(tup)

    x = 0.0
    y = 0.0

    # Records number of agreements and disagreements between pairs in the two vectors
    for elt in pairs_vec_one:

        if elt in pairs_vec_two:
            x = x + 1.0
        else:
            y = y + 1.0

    # Calculates the actual result
    result = (x - y) / (x + y)

    return result         

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
range_vector describes range of values for objective functions
format [f1_extreme_1, f1_extreme_2, f2_extreme_1, f2_extreme_2, etc...]
Note: ordering of extreme does matter
'''
def generate_data(id="Test",range_vector = [100, 150, 75, -50, 75, -50], num_points=50, noise=10):
    np.random.seed(101) 
    
    objective_values = []
    for i in range(int(len(range_vector)/2)):
        x = np.linspace(range_vector[i*2], range_vector[i*2+1], num_points)
        x += np.random.uniform(-1*(noise), noise, num_points) 
        objective_values.append(x)
        
    objective_value_tuples = [] 
    
    for i in range(num_points):
        tuple_list = []
        for objective_value_array in objective_values:
            tuple_list.append(objective_value_array[i])
        objective_value_tuples.append(tuple(tuple_list))
    
    plt.title("Objective Value Pairs " + str(id))
    plt.xlabel('weighted_value') 
    plt.ylabel('max_worst_case') 
    plt.scatter([x[0] for x in objective_value_tuples], [y[1] for y in objective_value_tuples], color="blue") 
    plt.show()

    return objective_value_tuples

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
    
    
    '''
    while len(sample_pairs) != 5:
        sample_pair = random.choice(objective_value_pairs)
        dominated = True
        while dominated == True:
            sample_pair = random.choice(objective_value_pairs)
            for pair in sample_pairs:
                if dominates(sample_pair, pair) == True:
                    sample_pairs.remove(pair)
                if dominates(sample_pair, pair) == False:
                    dominated == False
                if dominates(pair, sample_pair) == True:
                    dominated == True
            if dominated == False:
                sample_pairs.append(sample_pair)
    '''
    #sample_pairs_list = tuples_to_list(sample_pairs)
    
    return [objective_value_lists, sample_tuples, sample_tuples_list]

def graph_hyperplane(user_objective_values, objective_value_lists, func1_lower, func1_upper, func2_upper, func2_lower, num_points, alpha_learned, show=False):
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

def graph_alphas():
    return