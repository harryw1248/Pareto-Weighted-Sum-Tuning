import sys
import numpy as np
import matplotlib.pyplot as plt
import pwst_util as util 
from sample_user_rank import Sample_User
import application_data

alpha_plot_lines = []
error_plot_lines = []

def pareto_weighted_sum_tuning(objective_value_tuples, alpha_vector, tolerance_vector, margin_from_half, iteration_limit):
    """
    Runs Pareto-Weighted-Sum-Tuning on a simulated sample user/decision-maker
    Returns criteria weights
    """
    f = open("svm_rank/user_queries_train.dat", "w")
    f.close()

    user_virtual = Sample_User(alpha_vector, tolerance_vector)

    alpha_vectors_learned = []
    mean_alpha_vectors = [] 
    alpha_0_relative_errors = []

    data_subset = util.get_data_subset(objective_value_tuples, margin_from_half)
    sample_tuples = data_subset[1]

    iteration_number = 0
    while iteration_number < iteration_limit and len(objective_value_tuples) != 0:
        iteration_number += 1
        alpha_vector_learned = util.user_feedback(sample_tuples, user_virtual, iteration_number)
        objective_value_pairs = [x for x in objective_value_tuples if x not in sample_tuples]
        data_subset = util.get_data_subset(objective_value_pairs, margin_from_half)
        sample_tuples = data_subset[1]
        alpha_vectors_learned.append(alpha_vector_learned)
        mean_alpha_vector_learned = util.average_vectors(alpha_vectors_learned)

        alpha_0_relative_error = abs(alpha_vector[0] - mean_alpha_vector_learned[0]) / alpha_vector[0]
        alpha_0_relative_errors.append(alpha_0_relative_error)

        mean_alpha_vectors.append(mean_alpha_vector_learned)

        user_1 = Sample_User(mean_alpha_vector_learned, [0 for i in range(len(mean_alpha_vector_learned))])
        for example in sample_tuples:
            user_1.user_decision(example) 

        del user_1
        user_virtual.clear_user_history()
    
    alpha_plot_lines.append(mean_alpha_vectors)

    alpha_0_relative_errors = [x*100 for x in alpha_0_relative_errors]
    error_plot_lines.append(alpha_0_relative_errors)

    return mean_alpha_vectors[-1]