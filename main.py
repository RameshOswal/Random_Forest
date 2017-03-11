# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 01:22:26 2017

@author: Ramesh Oswal
"""
import numpy as np
import random_forest as rf
from scipy import stats
#Used this blog to understand the structure of Decision Tree making: http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html
def read_input(file_name):
    data = open(file_name)
    my_data = []
    for i in data.readlines():
        row = i.split(",")
        my_data.append(map(float,row[:-1])+[row[-1][:-1]])
    return np.array(my_data)[:,:-1],np.array(my_data)[:,-1]
def read_output(file_name):
    data = open(file_name)
    my_data = []
    for i in data.readlines():
        row = i.rstrip('\n').split(",")
        my_data.append(map(float,row[:-1]) + [row[-1]])
    return np.array(my_data)[:,:-1],np.array(my_data)[:,-1]
    
def train_test_split(folds,i):
    test = folds[i]
    train = []
    for i in (folds[:i] + folds[i + 1:]):
        for j in i:
            train += [j ]
    return train,test
from decision_tree import decision_tree

def decision_tree_function(trainX,trainY,testX,testY):
    dt = decision_tree() 
    dt.fit(trainX,trainY)
    s =  dt.predict(testX)
    print s
    print dt.acc_score(s,testY)

def random_forest_function(trainX,trainY,testX,testY):
    dt = rf.RandomForest(num_trees=3) 
    dt.fit(trainX,trainY)
    print testX.shape
    s =  dt.predict(testX)
    m = stats.mode(s)
#    print s
    print dt.acc_score(m[0][0],testY)
if __name__ == "__main__":
    trainX,trainY = read_input("train.csv")
    testX,testY = read_output("test.csv")
#    decision_tree_function(trainX,trainY,testX,testY)
    random_forest_function(trainX,trainY,testX,testY)
    
    
    
    
    
    
    
    
    
    
    
    
    
#    from simple_k_folds import k_fold_validation
#    obj = k_fold_validation()
#    obj.fit(my_data)
#    folds = obj.get_folds()
#    acc = []
#    for f in range(len(folds)):
#        # print len(folds)
#        dt = decision_tree()
#        train,test = train_test_split(folds, f)
#        # print len(train),len(test)
#        xTest,yTest  = test[:,:-1],test[:,-1]
#        # print xTest.shape,yTest.shape
#        tree = dt.buildtree(train)
#        # printtree(tree)
#        predicted_label = []
#        for i in test:
#            for key in dt.classify(i,tree ):
#                predicted_label.append(key)
#        acc.append( dt.acc_score(yTest,predicted_label))
#        print acc[-1]
#    import numpy as np
#    mean_acc = np.mean(acc)
#    std_acc = np.std(acc)
#    print mean_acc,std_acc