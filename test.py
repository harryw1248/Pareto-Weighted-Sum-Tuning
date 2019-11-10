import numpy as np
import matplotlib.pyplot as plt
import helper
import pandas as pd
from sklearn import svm
from sample_user import Sample_User

def generate_data():
    np.random.seed(101) 
    x = np.linspace(100, 150, 50) #weighted_value
    y = np.linspace(100, 50, 50)  #worst_case
  
    x += np.random.uniform(-6, 6, 50) 
    y += np.random.uniform(-6, 6, 50) 

    objective_value_pairs = [] 
    counter = 0
    for item in x:
        pair = (x[counter],y[counter])
        objective_value_pairs.append(pair)
        counter += 1

    return objective_value_pairs

def plot_decision_points(user_decisions, objective_value_pairs, slope=0, intercept=0):

    positive = []
    negative = []

    for decision in user_decisions:
        if user_decisions[decision] == 1:
            positive.append(decision)
        else:
            negative.append(decision)

    plt.title("User Decisions")
    plt.xlabel('weighted_value') 
    plt.ylabel('max_worst_case') 
    plt.scatter([x[0] for x in positive], [y[1] for y in positive], color="green") 
    plt.scatter([x[0] for x in negative], [y[1] for y in negative], color="red")

    xmin = min([x[0] for x in positive]+[x[0] for x in negative])
    xmax = max([x[0] for x in positive]+[x[0] for x in negative])
    num_points = 2 * len(positive)

    x = np.linspace(xmin, xmax, num_points) #weighted_value
    plt.plot(x, slope*x + intercept, linestyle='solid')
    plt.show()
    return plt

def online_feedback(objective_value_pairs):
    
    positive = []
    negative = []

    for i in range(24):
        pair_1 = objective_value_pairs[i]
        pair_2 = objective_value_pairs[49-i]
        print("Choices (first value is weighted_value, second is max_worst_case)")
        print("1: " + str(pair_1))
        print("2: " + str(pair_2))
        print("Type either 1 or 2")
        chosen = input()

        if chosen == "1":
            positive.append(pair_1)
            negative.append(pair_2)
        else:
            positive.append(pair_2)
            negative.append(pair_1)

        plt.xlabel('weighted_value') 
        plt.ylabel('max_worst_case') 
        plt.scatter([x[0] for x in positive], [y[1] for y in positive], color="green") 
        plt.scatter([x[0] for x in negative], [y[1] for y in negative], color="red")
        
        if i % 3 == 0:
            plt.show() 


"""
Takes in list of tuples and converts it to list of 2 element lists
"""
def tuples_to_list(pairs):
    new_list = []
    for item in pairs:
        sub_list = [item[0], item[1]]
        new_list.append(sub_list)
    
    return new_list

def extract_objective_value_pairs(df):
    objective_value_pairs = []
    alphas = []

    for alpha in df["user_alpha"]:
        alphas.append(alpha)
    
    counter = 0
    for weighted_value in df["weighted_value"]:
        pair = (weighted_value,df["max_worst_case"][counter])
        objective_value_pairs.append(pair)
        counter += 1
    
    return objective_value_pairs, alphas

def trial_1():
    #Note: normalize the datapoints themselves?
    objective_value_pairs = generate_data()
    user_1 = Sample_User(0.2, 0.07)

    for i in range(25):
        user_1.user_decision(objective_value_pairs[i],objective_value_pairs[49-i])

    user_decisions = user_1.get_user_decisions()
    objective_value_vectors = tuples_to_list(objective_value_pairs)
    plt = plot_decision_points(user_decisions, objective_value_pairs)

    decision_labels = []
    
    for pair in objective_value_pairs:
        decision = user_decisions[pair]
        decision_labels.append(decision)
    
    clf = svm.LinearSVC()
    clf.fit(objective_value_vectors, decision_labels)
    params = clf.coef_[0]
    print("SVM params")
    print(params)
    sum_params = sum(params)
    normalized_params = [(x/sum_params) for x in params]
    print("Normalized Parameters")
    print(normalized_params)

    w = clf.coef_[0]
    slope = -w[0] / w[1]
    intercept = clf.intercept_
    print("intercept")
    print(intercept)
    plt = plot_decision_points(user_decisions, objective_value_pairs, slope, intercept)

    '''
    x = np.linspace(100, 150, 50) #weighted_value
    # get the separating hyperplane
    w = clf.coef_[0]
    print("w:")
    print(w)
    a = w[1] / w[0]
    b = clf.intercept_[0]
    print("b:")
    print(b)
    plt.plot(x, a*x + b, linestyle='solid')

    plt.show()
    '''

def main():
    #objective_value_pairs = generate_data()
    #online_feedback(objective_value_pairs)
    trial_1()

    
    



main()

