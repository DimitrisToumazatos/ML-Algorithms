from id3 import *
import pandas as pd
from random import randint
import numpy as np
import pickle
from operator import add



class RandomForest:
    # n = 101, m = 40, PosAccuracy = 79 %, NegAccuracy = 38 %
    def __init__(self, n, m):
        self.nTrees = n
        self.mFeatures = m 
        self.trainingList = []
        self.trainingListResults = []


    def train(self, pos_filename, neg_filename):
        
        print("Training started")
        print("Reading positive training data...")

        df = pd.read_csv(pos_filename, header=None)
        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(1) # is positive


        print("Reading negative training data...")        

        df = pd.read_csv(neg_filename, header = None)
        for i in range(df.shape[0]):
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(0)  # is negative

        print("Creating Trees...")
        self.trainingList = np.array(self.trainingList)
        self.trainingListResults = np.array(self.trainingListResults)

        self.trees = []
        for i in range(self.nTrees):
            features = [randint(0, int((df.size/df.shape[0])-1)) for i in range(self.mFeatures)]
            self.trees.append(ID3(features))
            self.trees[i].fit(self.trainingList, self.trainingListResults) 

        print("Training finished")

                                                    
    def test(self, pos_filename, neg_filename):

        print("Test started")
        print("Testing positive test data...")
        
        df = pd.read_csv(pos_filename, header = None) # read positive examples
        totalPosTests = df.shape[0]
        testData = []
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
            temp = t.predict(testData) 
            temp = list(temp) 
            treeOut = list(map(add, treeOut, temp))

        finalPositive = 0
        finalNegative = 0

        count = 0
        for i in treeOut:                          # get the accuracy of the model
         
            if count < totalPosTests:
                if i > self.nTrees/2:
                    finalPositive += 1
            else:
                if i < self.nTrees:
                    finalNegative += 1
            count+=1

        print("Test finished")

        print(finalPositive)
        print(finalNegative)
        print(totalPosTests)
        print(totalNegTests)

        # print accuracy
        print("For the positive examples the accuracy is: " + str(round(((finalPositive/totalPosTests) * 100), 2)))   
        print("For the negative examples the accuracy is: " + str(round(((finalNegative/totalNegTests) * 100), 2)))  
        


myRandomForest = RandomForest(101, 40)                               # create object
myRandomForest.train("positive.csv", "negative.csv")                 # train model

print("Save training this time...")
        
with open('myRandomForest.pkl', 'wb') as outp:                       # save object 
    pickle.dump(myRandomForest, outp, pickle.HIGHEST_PROTOCOL) 

with open('myRandomForest.pkl', 'rb') as inp:                        # read saved object
    myRandomForest = pickle.load(inp)

myRandomForest.test("positiveDev.csv", "negativeDev.csv")            # test model


