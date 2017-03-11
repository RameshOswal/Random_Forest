# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:01:48 2017

@author: Ramesh Oswal
"""
from decision_tree import decision_tree
import numpy as np
class RandomForest:
    def __init__(self,num_trees=10,min_features_percentage=50,min_samples_percentage=50):
        self.models = []
        self.num_trees = num_trees
        self.min_features_percentage=float(min_features_percentage)
        self.min_samples_percentage=float(min_samples_percentage)
    def acc_score(self,a,b):
        return np.sum(a==b)/float(len(a))
    def fit(self,trainX,trainY):
        for i in range(self.num_trees):
            dt = decision_tree()
            feature_indexes = np.arange(len(trainX[0]))
            sample_indexes = np.arange(len(trainX))
            #suffle feature numbers
            np.random.shuffle(feature_indexes)
            np.random.shuffle(sample_indexes)
            #now pick random k and n features and sample indexes from our train
            feature_index_random_end = np.random.randint(1,len(trainX[0])-1)
            sliced_feature_indexes = feature_indexes[:feature_index_random_end]
            print "************",sliced_feature_indexes
            sampled_indexes_random_end = np.random.randint(1,len(trainX)-1)
            sliced_sampled_indexes = sample_indexes[:sampled_indexes_random_end]
            sliced_feature_indexes
            X = trainX[sliced_sampled_indexes ,: ]
            X = X[:,sliced_feature_indexes]
            Y = trainY[sliced_sampled_indexes]
            dt.fit(X,Y)
            self.models.append({'model':dt,'features':sliced_feature_indexes})
    def predict(self,testX):
            Y_predictions = []
            for model in self.models:
                Y_prediction = model['model'].predict(testX[:,model['features']])
                Y_predictions += [Y_prediction]
            Y_predictions =np.array(Y_predictions)
            prediction = Y_predictions
            return prediction