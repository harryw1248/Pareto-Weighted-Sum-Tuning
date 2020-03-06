import numpy as np
import matplotlib.pyplot as plt


def binomial_model(N, S0, u, r, K):
    """
    N = number of binomial iterations
    S0 = initial stock price
    u = factor change of upstate
    r = risk free interest rate per annum
    K = strike price
    """
    d = 1 / u
    p = (1 + r - d) / (u - d)
    q = 1 - p

    # make stock price tree
    stock = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            stock[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    print("stock")
    print(stock)

    '''
    # Generate option prices recursively
    option = np.zeros([N + 1, N + 1])
    option[:, N] = np.maximum(np.zeros(N + 1), (stock[:, N] - K))
    for i in range(N - 1, -1, -1):
        for j in range(0, i + 1):
            option[j, i] = (
                1 / (1 + r) * (p * option[j, i + 1] + q * option[j + 1, i + 1])
            )
    return option
    '''
    return stock

def generate_stock_objective_values():
    factor_change = 1.3
    price = 30
    objective_value_tuples = []

    for i in range(10):
        for j in range(20):
            stock_prices = binomial_model(2, price, factor_change, 0.25, 8)
            optimistic_gain = stock_prices[0][1] - price
            pessimistic_loss = stock_prices[1][1] - price
            objective_value_tuples.append((optimistic_gain,pessimistic_loss))
            factor_change += 0.01
        price += 0.01


    plt.title("Objective Value Pairs")
    plt.xlabel('Optimistic Gain') 
    plt.ylabel('Pessimistic Loss') 
    plt.scatter([x[0] for x in objective_value_tuples], [y[1] for y in objective_value_tuples], color="blue") 
    plt.show()

    return objective_value_tuples


if __name__ == "__main__":
    print("Calculating example option price:")
    #op_price = binomial_model(5, 4, 2, 0.25, 8)
    '''
    op_price = binomial_model(2, 5, 2, 0.25, 8)
    op_price = binomial_model(2, 5, 2.1, 0.25, 8)

    op_price = binomial_model(1, 40, 2, 0.25, 10)
    op_price = binomial_model(1, 40, 2, 0.6, 10)
    '''
    generate_stock_objective_values()

    #print(op_price)