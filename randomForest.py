from id3 import *
import pandas as pd
from random import randint
import numpy as np
import pickle
from operator import add

k=8000             # Number of examples in each training category

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

        count = 0
        for i in range(df.shape[0]):
            if count == k:
                break
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(1)              # is positive
            count += 1


        print("Reading negative training data...")
        df = pd.read_csv(neg_filename, header = None)       # read negative examples

        count = 0
        for i in range(df.shape[0]):
            if count == k:
                break
            row = [int(x) for x in df.iloc[i, :]]
            self.trainingList.append(row)
            self.trainingListResults.append(0)              # is negative
            count += 1

        print("Creating Trees...")

        self.trainingList = np.array(self.trainingList)
        self.trainingListResults = np.array(self.trainingListResults)

        for i in range(self.nTrees):

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

        truePositive = 0
        trueNegative = 0

        count = 0
        for i in treeOut:                          # get the accuracy of the model
            if count < totalPosTests:
                if i > self.nTrees/2:
                    truePositive += 1
            else:
                if i < self.nTrees:
                    trueNegative += 1
            count+=1

        print("Test finished")

        falsePositive = totalNegTests-trueNegative
        falseNegative = totalPosTests-truePositive

        # print train statistics
        print("True Positive: " + str(truePositive))
        print("False Positive: " + str(falsePositive))
        print("True Negative: " + str(trueNegative))
        print("False Negative: " + str(falseNegative))
        print("The accuracy for the positive data is: " + str(round(((truePositive/totalPosTests)), 3)))
        print("The accuracy for the negative data is: " + str(round(((trueNegative/totalNegTests)), 3)))
        print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(totalNegTests+totalPosTests))), 3)))  
        precision = round((truePositive/(truePositive+falsePositive)), 3)
        print("For the positive data the precision is: " + str(precision))  
        recall =  round((truePositive/(truePositive+falseNegative)), 3)
        print("For the positive data the recall is: " + str(recall))    
        print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))
        

    def testTrain(self, pos_filename, neg_filename):

        print("Test started")

        print("Testing positive test data...")
        
        testData = []

        df = pd.read_csv(pos_filename, header = None) # read positive examples
        totalPosTests = k

        count = 0
        for i in range(totalPosTests):
            if count == k:
                break
            row = [int(x) for x in df.iloc[i, :]]
            testData.append(row)
            count += 1

        print("Testing negative test data...")

        df = pd.read_csv(neg_filename, header = None) # read negative examples
        totalNegTests = k

        count =0
        for i in range(totalNegTests):
            if count == k:
                break
            row = [int(x) for x in df.iloc[i, :]]
            testData.append(row)
            count += 1

        print("Calculating output...")

        testData = np.array(testData)
        treeOut = [0] * 2*k            # predict the estimated results for each examples

        for t in self.trees:   
            temp = list(t.predict(testData))
            treeOut = list(map(add, treeOut, temp))

        truePositive = 0
        trueNegative = 0

        count = 0
        for i in treeOut:                          # get the accuracy of the model
            if count < totalPosTests:
                if i > self.nTrees/2:
                    truePositive += 1
            else:
                if i < self.nTrees:
                    trueNegative += 1
            count+=1

        print("Test finished")

        falsePositive = totalNegTests-trueNegative
        falseNegative = totalPosTests-truePositive

        # print train statistics
        print("True Positive: " + str(truePositive))
        print("False Positive: " + str(falsePositive))
        print("True Negative: " + str(trueNegative))
        print("False Negative: " + str(falseNegative))
        print("The accuracy for the positive data is: " + str(round(((truePositive/totalPosTests)), 3)))
        print("The accuracy for the negative data is: " + str(round(((trueNegative/totalNegTests)), 3)))
        print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(totalNegTests+totalPosTests))), 3)))  
        precision = round((truePositive/(truePositive+falsePositive)), 3)
        print("For the positive data the precision is: " + str(precision))  
        recall =  round((truePositive/(truePositive+falseNegative)), 3)
        print("For the positive data the recall is: " + str(recall))    
        print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))
        

myRandomForest = RandomForest(101, 4)                               # create object
myRandomForest.train("positiveTrain.csv", "negativeTrain.csv")                 # train model


print("Save model...")
"""
with open('myRandomForest-301-40.pkl', 'wb') as outp:            # save object 
    pickle.dump(myRandomForest, outp, pickle.HIGHEST_PROTOCOL) 

with open('myRandomForest-301-40-20k.pkl', 'rb') as inp:             # read saved object
    myRandomForest = pickle.load(inp)
"""

myRandomForest.test("positiveTest.csv", "negativeTest.csv")          # test model





