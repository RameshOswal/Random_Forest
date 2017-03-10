# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 01:22:26 2017

@author: Ramesh Oswal
"""
#Used this blog to understand the structure of Decision Tree making: http://www.patricklamle.com/Tutorials/Decision%20tree%20python/tuto_decision%20tree.html
def read_input(file_name):
    data = open(file_name)
    my_data = []
    for i in data.readlines():
        row = i.split(",")
        my_data.append(map(float,row[:-1])+[row[-1][:-1]])
    return my_data
def read_output(file_name):
    data = open(file_name)
    my_data = []
    for i in data.readlines():
        row = i.split(",")
        my_data.append(map(float,row[:-1]))
    return my_data
def train_test_split(folds,i):
    test = folds[i]
    train = []
    for i in (folds[:i] + folds[i + 1:]):
        for j in i:
            train += [j ]
    return train,test
from decision_tree import decision_tree
if __name__ == "__main__":
    my_data = read_input("train.csv")
    my_test = read_output("test.csv")
    output = open("labels.txt","w+")
    from simple_k_folds import k_fold_validation
    obj = k_fold_validation()
    obj.fit(my_data)
    folds = obj.get_folds()
    acc = []
    for f in range(len(folds)):
        # print len(folds)
        dt = decision_tree()
        train,test = train_test_split(folds, f)
        # print len(train),len(test)
        xTest,yTest  = test[:,:-1],test[:,-1]
        # print xTest.shape,yTest.shape
        tree = dt.buildtree(train)
        # printtree(tree)
        predicted_label = []
        for i in test:
            for key in dt.classify(i,tree ):
                predicted_label.append(key)
        acc.append( dt.acc_score(yTest,predicted_label))
        print acc[-1]
    import numpy as np
    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    print mean_acc,std_acc