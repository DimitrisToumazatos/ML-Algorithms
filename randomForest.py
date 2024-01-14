from id3 import *
from random import randint
import numpy as np
from operator import add

k=8000             # Number of examples in each training category

class RandomForest:

    def __init__(self, n, m):
        #The init of the rf sets the
        #number of trees created and the
        #number of features used
        self.nTrees = n                 # number of trees
        self.mFeatures = m              # number of features        
        self.trees = []                 # list containing the trees

    def train(self, x_train, y_train, columns):
        #We create the trees with the m random features
        #using the id3 algorithm
        print("Training started")
        print("Creating Trees...")

        self.trainingList = np.array(x_train)   # list containing the vectors for the training examples
        self.trainingListResults = np.array(y_train)    # list containing the correct category for each of the above examples

        for i in range(self.nTrees):

            features = [randint(0, int((columns)-1)) for j in range(self.mFeatures)]        # selects the m random features
            self.trees.append(ID3(np.array(features)))      #creates the new tree
            self.trees[i].fit(self.trainingList, self.trainingListResults) 
            print("Tree number " + str(i) + " has been created!")

        print("Training finished")

                                                    
    def test(self, testData):

        print("Test started")

        print("Calculating output...")

        treeOut = [0] * len(testData)       # predict the estimated results for each examples
        testData = np.array(testData)       #convering the testData list to np array
        
        for t in self.trees:   
            temp = list(t.predict(testData))    #returns a list with the predicted results of a tree
            treeOut = list(map(add, treeOut, temp))     #adds the results of the trees for the testData

        rfPositive = 0
        rfNegative = 0

        count = 0
        for i in treeOut:                          # gets the accuracy of the model
            if count < round(len(testData) / 2):    #Τουμ βαλε εδώ σχόλια δε θυμάμαι καλά
                if i > self.nTrees/2:
                    rfPositive += 1
            else:
                if i < self.nTrees:
                    rfNegative += 1
            count+=1

        print("Test finished")

        return rfPositive, rfNegative