from id3 import *
import pandas as pd
from random import randint
import numpy as np
import pickle
from operator import add



class RandomForest:
    # n = 301, m = 40, PosAccuracy = 76 %, NegAccuracy = 100 %
    # n = 301, m = 50, PosAccuracy = 77 %, NegAccuracy = 100 %
    def __init__(self, n, m):
        self.nTrees = n                 # number of trees
        self.mFeatures = m              # number of features
        self.trainingList = []          # list containing the vectors for the training examples
        self.trainingListResults = []   # list containing the correct category for each of the above examples
        self.trees = []                 # list containing the trees

    def train(self, pos_filename, neg_filename):
        
        print("Training started")


        print("Reading positive training data...")
        df = pd.read_csv(pos_filename, header=None)         # read positive examples

        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(1)              # is positive


        print("Reading negative training data...")
        df = pd.read_csv(neg_filename, header = None)       # read negative examples

        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(0)              # is negative


        print("Creating Trees...")

        self.trainingList = np.array(self.trainingList)
        self.trainingListResults = np.array(self.trainingListResults)

        count = 0
        for i in range(self.nTrees):
            print("Tree: " + str(count))                    # print the count of trained trees
            count+=1

            features = [randint(0, int((df.shape[1])-1)) for i in range(self.mFeatures)]        # create the tree
            self.trees.append(ID3(np.array(features)))
            self.trees[i].fit(self.trainingList, self.trainingListResults) 
            print("Tree number " + str(i) + " has been created!")

        print("Training finished")

                                                    
    def test(self, pos_filename, neg_filename):

        print("Test started")

        print("Testing positive test data...")
        
        testData = []

        df = pd.read_csv(pos_filename, header = None) # read positive examples
        totalPosTests = df.shape[0]

        for i in range(totalPosTests):
            row = [int(x) for x in df.iloc[i, :]]
            testData.append(row)


        print("Testing negative test data...")

        df = pd.read_csv(neg_filename, header = None) # read negative examples
        totalNegTests = df.shape[0]

        for i in range(totalNegTests):
            row = [int(x) for x in df.iloc[i, :]]
            testData.append(row)

        print("Calculating output...")

        testData = np.array(testData)
        treeOut = [0] * testData.shape[0]            # predict the estimated results for each examples

        for t in self.trees:   
            temp = list(t.predict(testData))
            treeOut = list(map(add, treeOut, temp))

        finalPositive = 0
        finalNegative = 0

        count = 0
        for i in treeOut:                          # get the accuracy of the model
            print(treeOut)
            if count < totalPosTests:
                if i > self.nTrees/2:
                    finalPositive += 1
            else:
                if i < self.nTrees:
                    finalNegative += 1
            count+=1

        print("Test finished")

        print("True Positive: " + str(finalPositive))                           # F1 Score
        print("False Positive: " + str(totalNegTests-finalNegative))
        print("True Negative: " + str(finalNegative))
        print("False Negative: " + str(totalPosTests-finalPositive))

        # print accuracy
        print("For the positive examples the accuracy is: " + str(round(((finalPositive/totalPosTests) * 100), 2)))   
        print("For the negative examples the accuracy is: " + str(round(((finalNegative/totalNegTests) * 100), 2)))  
        

myRandomForest = RandomForest(301, 40)                               # create object
myRandomForest.train("positive.csv", "negative.csv")                 # train model

print("Save model...")

with open('myRandomForest-301-40-16k.pkl', 'wb') as outp:            # save object 
    pickle.dump(myRandomForest, outp, pickle.HIGHEST_PROTOCOL) 
"""
with open('myRandomForest-301-40-16k.pkl', 'rb') as inp:             # read saved object
    myRandomForest = pickle.load(inp)
"""

myRandomForest.test("positiveTest.csv", "negativeTest.csv")          # test model


