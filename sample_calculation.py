import pickle 
import multi_objective_reccomender_system as alpha_rank
from statistics import mean 
from statistics import stdev
from math import sqrt
import matplotlib.pyplot as plt


def store_data(flag, plot_lines):
    dbfile_0 = None
    db_0 = None 

    dbfile_1 = None
    db_1 = None

    dbfile_2 = None
    db_2 = None

    dbfile_3 = None
    db_3 = None

    if flag == 'lower_values':
        dbfile_0 = open('pickle_1', 'rb') 
        db_0 = pickle.load(dbfile_0) 

        dbfile_1 = open('pickle_2', 'rb') 
        db_1 = pickle.load(dbfile_1) 

        dbfile_2 = open('pickle_3', 'rb') 
        db_2 = pickle.load(dbfile_2) 

        dbfile_3 = open('pickle_4', 'rb') 
        db_3 = pickle.load(dbfile_3) 

        dbfile_0.close() 
        dbfile_1.close()
        dbfile_2.close() 
        dbfile_3.close()
        dbfile_0 = open('pickle_1', 'wb') 
        dbfile_1 = open('pickle_2', 'wb') 
        dbfile_2 = open('pickle_3', 'wb') 
        dbfile_3 = open('pickle_4', 'wb')         
    else:
        dbfile_0 = open('pickle_5', 'rb') 
        db_0 = pickle.load(dbfile_0) 

        dbfile_1 = open('pickle_6', 'rb') 
        db_1 = pickle.load(dbfile_1) 

        dbfile_2 = open('pickle_7', 'rb') 
        db_2 = pickle.load(dbfile_2) 

        dbfile_3 = open('pickle_8', 'rb') 
        db_3 = pickle.load(dbfile_3) 

        dbfile_0.close() 
        dbfile_1.close()
        dbfile_2.close() 
        dbfile_3.close()
        dbfile_0 = open('pickle_5', 'wb') 
        dbfile_1 = open('pickle_6', 'wb') 
        dbfile_2 = open('pickle_7', 'wb') 
        dbfile_3 = open('pickle_8', 'wb')   

    for i in range(4):
        if i == 0:
            db_0.append(plot_lines[i])
        elif i == 1:
            db_1.append(plot_lines[i])
        elif i == 2:
            db_2.append(plot_lines[i])
        else:
            db_3.append(plot_lines[i])

    print("db_0: " + str(db_0))
    print("db_1: " + str(db_1))
    print("db_2: " + str(db_2))
    print("db_3: " + str(db_3))

    pickle.dump(db_0, dbfile_0)
    pickle.dump(db_1, dbfile_1) 
    pickle.dump(db_2, dbfile_2) 
    pickle.dump(db_3, dbfile_3)                       
    dbfile_0.close() 
    dbfile_1.close()
    dbfile_2.close() 
    dbfile_3.close()

    return db_0, db_1, db_2, db_3

def main():


    pickle_out_1 = open('pickle_1','wb')
    pickle_out_2 = open('pickle_2','wb')
    pickle_out_3 = open('pickle_3','wb')
    pickle_out_4 = open('pickle_4','wb')

    pickle_out_5 = open('pickle_5','wb')
    pickle_out_6 = open('pickle_6','wb')
    pickle_out_7 = open('pickle_7','wb')
    pickle_out_8 = open('pickle_8','wb')

    pickle.dump([], pickle_out_1)
    pickle_out_1.close()
    pickle.dump([], pickle_out_2)
    pickle_out_2.close()
    pickle.dump([], pickle_out_3)
    pickle_out_3.close()
    pickle.dump([], pickle_out_4)
    pickle_out_4.close()

    pickle.dump([], pickle_out_5)
    pickle_out_5.close()
    pickle.dump([], pickle_out_6)
    pickle_out_6.close()
    pickle.dump([], pickle_out_7)
    pickle_out_7.close()
    pickle.dump([], pickle_out_8)
    pickle_out_8.close()

    margins_from_half = [1,2,3,4]
    db_0s = [[] for i in range(15)]
    db_1s = [[] for i in range(15)]
    db_2s = [[] for i in range(15)]
    db_3s = [[] for i in range(15)]

    for i in range(10):
        plot_lines = alpha_rank.experiment(margins_from_half)
        #db_0, db_1, db_2, db_3 = store_data('lower_values', plot_lines)
        for i in range(15):
            '''
            db_0s.append(db_0[i])
            db_1s.append(db_1[i])
            db_2s.append(db_2[i])
            db_3s.append(db_3[i])
            '''
            db_0s[i].append(plot_lines[0][i])
            db_1s[i].append(plot_lines[1][i])
            db_2s[i].append(plot_lines[2][i])
            db_3s[i].append(plot_lines[3][i])
    
    db_0s_mean = [mean(i) for i in db_0s]
    db_1s_mean = [mean(i) for i in db_1s]
    db_2s_mean = [mean(i) for i in db_2s]
    db_3s_mean = [mean(i) for i in db_3s]
    db_means = [db_0s_mean, db_1s_mean, db_2s_mean, db_3s_mean]
    print(db_means)

    db_0s_intervals = [stdev(i)/sqrt(10) for i in db_0s]
    db_1s_intervals = [stdev(i)/sqrt(10) for i in db_0s]
    db_2s_intervals = [stdev(i)/sqrt(10) for i in db_0s]
    db_3s_intervals = [stdev(i)/sqrt(10) for i in db_0s]

    print("db_0s_intervals: " + str(db_0s_intervals))
    print("db_1s_intervals: " + str(db_1s_intervals))
    print("db_2s_intervals: " + str(db_2s_intervals))
    print("db_3s_intervals: " + str(db_3s_intervals))

    title = "Alpha Relative Error Progress (Random Sampling)"
    plt.title(title)
    plt.xlabel("Iteration Number") 
    plt.ylabel("Alpha Error Percentage")
    for i in range(len(margins_from_half)):
        relative_error_plot_name = str(margins_from_half[i]*2+1) + " point sampling"
        relative_error_plot = plt.plot([i for i in range(15)], db_means[i], label=relative_error_plot_name)
    plt.legend()
    plt.show()

    margins_from_half = [5,6,7,8]
    plot_lines = alpha_rank.experiment(margins_from_half)
    store_data('lower_values', plot_lines)

    

def sample_calculate():
    #lower
    margins_from_half = [1,2,3,4]
    lower_vector = [[4.534441350700506, 1.526942537643683, 1.6548331117701525, 1.9022069774259354, 2.083450178613222, 1.56694600758119, 1.2917759385201455, 1.3612683809265207, 1.085461529782931, 1.1976318838956002, 1.1680233113352534, 1.2843652620429895, 1.2931739327649017, 1.3233905083553563, 1.2843922845841758], [3.050784712313533, 3.5856269367782008, 3.426163888483454, 3.3314147167719153, 3.1188267228897857, 3.0382757477343354, 3.0964079963575384, 2.9230690213255346, 2.723672717871791, 2.594721114435243, 2.4831329867561944, 2.327725178800459, 2.237158772120644, 2.182579535319048, 2.119455479646741], [1.8060917146904092, 1.114291785323222, 0.6931594680710251, 0.4471255850811984, 0.3091878918936401, 0.22275782636120106, 0.03954796584602442, 0.23672362370654362, 0.39665907788044574, 0.5516841817395461, 0.5455209645460387, 0.6042429126599247, 0.6944838510350437, 0.800871005690836, 0.8105198904064064], [0.8023746714688196, 0.8594992691413617, 0.5927019364774615, 0.47088297293601533, 0.34681654306497245, 0.24094730822197635, 0.000813949656230939, 0.04733700566263618, 0.2529329447373274, 0.3501045252721954, 0.4825098979974185, 0.5547321023070725, 0.6793138229110689, 0.7913059221907142, 0.7651691284343842]]
    title = "Alpha Relative Error Progress (Random Sampling)"
    plt.title(title)
    plt.xlabel("Iteration Number") 
    plt.ylabel("Alpha Error Percentage")
    for i in range(len(margins_from_half)):
        relative_error_plot_name = str(margins_from_half[i]*2+1) + " point sampling"
        relative_error_plot = plt.plot([i for i in range(15)], lower_vector[i], label=relative_error_plot_name)
    plt.legend()
    plt.show()

sample_calculate()

