from id3 import *
from random import randint
import numpy as np
from operator import add

class RandomForest:

    # Constructor.
    # Sets up the Random Forest Object
    def __init__(self, n, m):
        self.nTrees = n                 # set the number of trees
        self.mFeatures = m              # set the number of features        
        self.trees = []                 # set the list containing the trees
    
    # Training Function.
    # Create n trees with m random features
    # using the id3 algorithm
    def train(self, x_train, y_train):
        print("Training started")
        print("Creating Trees...")

        x_train = np.array(x_train)           # list containing the vectors for the training examples.
        y_train = np.array(y_train)           # list containing the correct category for each of the above examples.

        numberOfFeatures = len(x_train[0])

        for i in range(self.nTrees):            # Create the trees

            features = [randint(0, int((numberOfFeatures)-1)) for j in range(self.mFeatures)]         # Select m random features
            self.trees.append(ID3(np.array(features)))                                                # and create a new tree.
            self.trees[i].fit(x_train, y_train)                                                       # Train the new tree.
            print("Tree number " + str(i) + " has been created!")

        print("Training finished")

    # Test the User's input.                                
    def test(self, testData):

        print("Test started")

        print("Calculating output...")

        treeOut = [0] * len(testData)                   # Create a list for the trees's predictions.
        testData = np.array(testData)                   # Convey the testData list to np array.
        
        for tree in self.trees:   
            temp = list(tree.predict(testData))         # Create a list with the predicted results of a tree
                                                        # for every given example.
            treeOut = list(map(add, treeOut, temp))     # adds the results of the trees.

        prediction = []
        for i in treeOut:                               # Make predictions for each example.
            if sum(i) > self.nTrees/2:                  # The example is deemed as positive.
                prediction.append(1)
            else:                                       # The example is deemed as negative.
                prediction.append(0)

        print("Test finished")

        return prediction