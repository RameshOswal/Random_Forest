# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 01:00:22 2017

@author: Ramesh Oswal
"""
import numpy as np
#Class to handle the tree node details for the decision tree node
class decision_tree:
    class decisionnode:
        def __init__(self, feature=-1, value=None, results=None, tb=None, fb=None,ent = 0):
            self.feature = feature #feature number on which the split will occur
            self.value = value #value to consider for condition on above feature number
            self.results = results #if its a leaf node then the resul of the class is stored here
            self.entropy = ent #entropy of the labels for given split
            self.tb = tb #link to node for true condition of above split
            self.fb = fb #link to node for false condition of above split
    
    # Divides a set on a specific column. Can handle numeric
    # or nominal values
    def splitting_feature(self,rows, column, value):
        # Make a function that tells us if a row is in
        # the first group (true) or the second group (false)
        split_function = None #split function based on categorical or continuous dataset
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value
        # based on the spliting function divides the labels into two sets
        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return (set1, set2)
    
    def __init__(self):
        self.tree = []
    # Create counts of possible results (the last column of
    # each row is the result)
    def uniquecounts(self,rows):
        results = {}
        for row in rows:
            # The result is the last column
            r = row[len(row) - 1]
            if r not in results: results[r] = 0
            results[r] += 1
        return results
    
    # Entropy is the sum of p(x)log(p(x)) across all
    # the different possible results
    def entropy(self,rows):
        from math import log
        log2 = lambda x: log(x) / log(2)
        results = self.uniquecounts(rows)
        # Now calculate the entropy
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * log2(p)
        return ent

    def printtree(self,tree,indent=''):

        # node is a leaf node
        if tree.results != None:
            print str(tree.results)
        else:
            print "feature_number" + str(tree.feature) + ':' + str(tree.value) + " Entropy =" + "{0:.2f}".format(tree.entropy) +  '? '
            print indent + 'T->',
            self.printtree(tree.tb, indent + '  ')
            print indent + 'F->',
            self.printtree(tree.fb, indent + '  ')
    
    #the predic function which uses the observation and tree struction o classify the observation to appropriate label
    def classify(self,observation, tree):
        if tree.results != None:
            return tree.results
        else:
            v = observation[tree.feature]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)
    
    def buildtree(self,rows):
        best_information_gain = 0.0
        best_criteria = None
        best_sets = None
        if len(rows) == 0: return self.decisionnode()
        current_score = self.entropy(rows)
        column_count = len(rows[0]) - 1
        for col in range(0, column_count):
            column_values = {}
            for row in rows:
                column_values[row[col]] = 1
            for value in column_values.keys():
                (set1, set2) = self.splitting_feature(rows, col, value)
                #Calculate the Information gain
                p = float(len(set1)) / len(rows)
                gain = current_score - p * self.entropy(set1) - (1 - p) * self.entropy(set2)
                if gain > best_information_gain and len(set1) > 0 and len(set2) > 0:
                    best_information_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        # Create the sub branches
        if best_information_gain > 0:
            trueBranch = self.buildtree(best_sets[0])
            falseBranch = self.buildtree(best_sets[1])
            return self.decisionnode(feature=best_criteria[0], value=best_criteria[1],
                                tb=trueBranch, fb=falseBranch,ent = self.entropy(rows) )
        else:
            return self.decisionnode(results=self.uniquecounts(rows))
    def acc_score(self,a,b):
        return np.sum(a==b)/float(len(a))
    def fit(self,trainX,trainY):
        #train  = trainX + trainY
        if type(trainX) != np.ndarray:
               trainX= np.array(trainX) 
        if type(trainX) != np.ndarray :
            trainY = np.array(trainY)
#        convert trainY into column vector
        trainY = trainY[np.newaxis].T
        train = np.concatenate((trainX,trainY),axis = 1)
        self.tree = self.buildtree(train)
    def predict(self,X):
        Y = np.array([])
        for x in X:
            Y = np.append(Y,[self.classify(x,self.tree).keys()[0]])
        return Y
    
