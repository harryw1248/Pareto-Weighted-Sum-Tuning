import os

os.system("./svm_rank_learn -c 3 train.dat model ")
os.system("./svm_rank_classify test.dat model predictions")