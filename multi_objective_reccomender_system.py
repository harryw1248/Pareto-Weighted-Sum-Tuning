import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import helper
import pandas as pd
from sklearn.linear_model import LinearRegression
from sample_user_rank import Sample_User

trials_data = []

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

"""
Takes in list of tuples and converts it to list of 2 element lists
"""
def tuples_to_list(pairs):
    new_list = []
    for item in pairs:
        sub_list = [item[0], item[1]]
        new_list.append(sub_list)
    
    return new_list

def trial(id='Test', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=10):

    trial_data = {'alpha_true': alpha, 'tolerance': tolerance, 'func1_lower': func1_lower, 'func1_upper': func1_upper, 
                 'func2_upper': func2_upper, 'func2_lower': func2_lower, 'num_points': num_points, 'noise': noise, 
                 'weight1': 0, 'weight2': 0, 'alpha_learned': 0}

    print("Trial " + str(id))
    objective_value_pairs = generate_data(id, func1_lower, func1_upper, func2_upper, func2_lower, num_points, noise)
    num_data_points = len(objective_value_pairs)
    half_point = int(num_data_points / 2)
    margin_from_half = int(num_data_points / 20)
    objective_value_lists = tuples_to_list(objective_value_pairs)
    user_1 = Sample_User(alpha, tolerance)

    sample_pairs = []
    sample_pairs_list = []
    for i in range(half_point-margin_from_half,half_point+margin_from_half):
        sample_pairs.append(objective_value_pairs[i])
        sample_pairs_list.append(objective_value_lists[i])

    for example in sample_pairs:
        user_1.user_decision(example)
    
    user_1.get_rankings()
    user_rank_indices = user_1.get_user_rank_indices()
    
    user_rank_indices_list = []
    for pair in sample_pairs:
        user_rank_indices_list.append(user_rank_indices[pair])
    
    reg = LinearRegression().fit(sample_pairs, user_rank_indices_list)
    score = reg.score(sample_pairs, user_rank_indices_list)
    coef = reg.coef_
    print("Coefficients:")
    print(coef)
    trial_data['weight1'] = coef[0] 
    trial_data['weight2'] = coef[1] 
    sum_coef = sum(coef)

    print("Normalized Coefficients:")
    print([x/sum_coef for x in coef])

    trial_data['alpha_learned'] = (coef[0] / sum_coef) 
    trial_data['alpha_error'] = ( abs((coef[0] / sum_coef) - alpha) / alpha ) 

    print("\n")
    trials_data.append(trial_data)

    for example in objective_value_pairs:
        user_1.user_decision(example)

    user_1.get_rankings()
    user_rank_indices = user_1.get_user_rank_indices()

    x_values = [x[0] for x in objective_value_lists]
    y_values = [y[1] for y in objective_value_lists]
    z_values = []

    for i in range(len(x_values)):
        key = (x_values[i], y_values[i])
        value = user_rank_indices[key]
        z_values.append(value)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x_values, y_values, z_values, cmap='Greens')
    ax.set_xlabel('Weighted Value')
    ax.set_ylabel('Max Worst Case')
    ax.set_zlabel('Multi-Objective Value')
    #plt.show()

    def f(alpha, x, y):
        return alpha*x + (1-alpha)*y
    
    x = np.linspace(func1_lower, func1_upper, num_data_points)
    y = np.linspace(func2_upper, func2_lower, num_data_points)
    X, Y = np.meshgrid(x, y)
    Z = f((coef[0]/sum_coef), X, Y)

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()
    
    return trial_data
    

def main():

    
    trial(id='0.0 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=10)
    #trial(id='0.1 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=15)
    #trial(id='0.2 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=20)
    #trial(id='0.3 50 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=50, noise=25)

    '''
    trial(id='0.0 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=10)
    trial(id='0.1 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=15)
    trial(id='0.2 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=20)
    trial(id='0.3 100 points', alpha=0.3, tolerance=0.05, func1_lower=100, func1_upper=150, func2_upper=100, func2_lower=50, num_points=100, noise=25)

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


main()

