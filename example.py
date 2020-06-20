"""
Example of using Pareto-Weighted-Sum-Tuning
A sample user is created and makes decisions on generated stock data.
PWST will output the sample user's criteria weight(s).
"""
import pareto_weighted_sum_tuning as pwst
import matplotlib.pyplot as plt
import application_data

def main():
    
    objective_value_tuples = application_data.generate_stock_objective_values()

    alpha_vector = [0.3]
    tolerance_vector = [0.05]

    points_per_iteration = [11, 13, 15, 17]

    iteration_limit = 15
        
    for setting in points_per_iteration:
        pwst.pareto_weighted_sum_tuning(objective_value_tuples, alpha_vector, tolerance_vector, setting, iteration_limit)

    title = "Alpha Relative Error Progress"
    plt.title(title)
    plt.xlabel("Iteration Number") 
    plt.ylabel("Alpha Error Percentage")


    for i in range(len(points_per_iteration)):
        relative_error_plot_name = str(points_per_iteration[i]) + " point sampling"
        plt.plot([i for i in range(iteration_limit)], pwst.plot_lines[i], label=relative_error_plot_name)

    plt.legend()
    plt.show()
    return pwst.plot_lines

if __name__ == "__main__":
    main()