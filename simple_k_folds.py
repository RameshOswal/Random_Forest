# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:32:00 2017

@author: Ramesh Oswal
"""

class k_fold_validation:
    def create_k_folds(self):
        if self.k>self.length:
            print "Error occured number of folds should be less than length"
            return "Error 1"
        else: 
            fold_length = int(self.length/self.k)
            fold_ranges = []
            for i in range(self.k):
                fold_ranges += [range(fold_length*i,fold_length*i + fold_length)]
            if self.length%self.k !=0:
                fold_ranges[-1] = fold_ranges[-1] + range(fold_ranges[-1][-1]+1,fold_ranges[-1][-1]+1+ (self.length%self.k))
            self.fold_ranges =  fold_ranges
            return fold_ranges
    def __init__(self,k=10):
        self.k=k
        self.data = []
        self.length = len(self.data)
        self.fold_ranges = []
        self.folds = []
    def fit(self,data):
        import numpy as np
        self.data = np.array(data)
        self.length = len(self.data)
        self.create_k_folds()
        print self.fold_ranges
        for ranges in self.fold_ranges:
            self.folds.append(self.data[ranges])
    def get_folds(self):
        return self.folds

