import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

########## Read Train Data ##########

posTrainData = []
negTrainData = []

#Reading of the Positive Training Data
print("Reading the positive training data!")

dfPos = pd.read_csv("positiveTrain.csv", header = None)     # read the positive training examples
sizePos = dfPos.shape[0]        # shape[0] contains the number of rows of our file

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    posTrainData.append(row)

#Reading of the Negative Training Data
print("Reading of the negative training data!")

dfNeg = pd.read_csv("negativeTrain.csv", header = None)     # read the negative training examples
sizeNeg = dfNeg.shape[0]        # shape[0] contains the number of rows of our file

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    negTrainData.append(row)

trainData = copy.deepcopy(posTrainData)
resultsTrain = []

for row in negTrainData:
    trainData.append(row)

for i in range(len(posTrainData)):
    resultsTrain.append(1)
for i in range(len(negTrainData)):
    resultsTrain.append(0)



######### Read Test Data ############
        
posTestData = []
negTestData = []

#Reading of the Positive Test Data
print("Reading the positive test data!")

dfPos = pd.read_csv("positiveDev.csv", header = None)     # read the positive test examples
sizePos = dfPos.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    posTestData.append(row)

#Reading of the Negative Test Data
print("Reading the negative test data!")

dfNeg = pd.read_csv("negativeDev.csv", header = None)     # read the negative test examples
sizeNeg = dfNeg.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    negTestData.append(row)


testData = copy.deepcopy(posTestData)
for row in negTestData:
    testData.append(row)

resultsTest = []
numberOfPositive = len(posTestData)
numberOfNegative = len(negTestData)

for i in range(numberOfPositive):
    resultsTest.append(1)
for i in range(numberOfNegative):
    resultsTest.append(0)




X_train, X_test, y_train, y_test = trainData, testData, resultsTrain, resultsTest
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (y_test != y_pred).sum()))