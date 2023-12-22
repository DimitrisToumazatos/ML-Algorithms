from id3 import *
import pandas as pd
from math import add
from random import randint

class randomForest:
    def __init__(self, n, m):
        self.nTrees = n
        self.mFeatures = m 
        self.trainingList = []
        self.trainingListResults = []


    def train(self, pos_filename, neg_filename):

        df = pd.read_csv(pos_filename, header=None)
        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(1) # is positive

        df = pd.read_csv(neg_filename, header = None)
        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(0)  # is negative


        self.trees = []
        for i in range(self.nTrees):
            features = [randint(0, (df.size/df.shape[0])-1) for i in range(self.mFeatures)]
            self.trees.append(ID3(features))
            self.trees[i].fit(self.trainingList, self.trainingListResults)    
                                                    
    def test(self, testData):
        
        
        treeOut = []
        for t in self.trees:
            treeOut.append(t.predict(testData))
        
        results = [0] * len(treeOut[0])
        for i in treeOut:
            count = 0
            for j in i:
                results[count] += j
                count += 1
            if results[count] > self.nTrees/2:
                print("positive")
            else:
                print("negative")


