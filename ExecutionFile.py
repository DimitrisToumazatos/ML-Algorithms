from Bayes import *
from AdaBoost import *
from randomForest import *
import pandas as pd
import copy

posTrainData = []
negTrainData = []
#Reading of the Positive Training Data
print("Reading of the positive training data!")

dfPos = pd.read_csv("positiveTrain.csv", header = None)     # read the positive training examples
sizePosTrain = dfPos.shape[0]        # shape[0] contains the number of rows of our file
colPos = dfPos.shape[1]         # shape[1] contains the number of columns of our file

for i in range(sizePosTrain):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that i shows 
    posTrainData.append(row)

#Reading of the Negative Training Data
print("Reading of the negative training data!")

dfNeg = pd.read_csv("negativeTrain.csv", header = None)     # read the negative training examples
sizeNegTrain = dfNeg.shape[0]        # shape[0] contains the number of rows of our file
colNeg = dfNeg.shape[1]         # shape[1] contains the number of columns of our file

for i in range(sizeNegTrain):
    row = [int(x) for x in dfNeg.iloc[i, : ]]       # df.iloc function returns the row of our csv that i shows
    negTrainData.append(row)

#Bayes initialization and Training
nB = NaiveBayes()
nB.train(posTrainData, colPos, negTrainData, colNeg)

#Train Data modifications
Train_Data = copy.deepcopy(posTrainData)    #creates a deepcopy of the posTrainData
for row in negTrainData:
    Train_Data.append(row)
results = []
for i in range(len(posTrainData)):
    results.append(1)
for i in range(len(negTrainData)):
    results.append(0)

#Random forest initialization and Training
rf = RandomForest(101, 4)
rf.train(Train_Data, results, colNeg)

#AdaBoost initialization and training
Ada = AdaBoost(100)
Ada.fit(Train_Data, results)



posTestData = []
negTestData = []
#Reading of the Positive Test Data
print("Reading of the positive test data!")

dfPos = pd.read_csv("positiveTest.csv", header = None)     # read the positive test examples
sizePosTest = dfPos.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizePosTest):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    posTestData.append(row)

#Reading of the Negative Test Data
print("Reading of the negative test data!")

dfNeg = pd.read_csv("negativeTest.csv", header = None)     # read the negative test examples
sizeNegTest = dfNeg.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizeNegTest):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    negTestData.append(row)

#Test Data modifications
testData = copy.deepcopy(posTestData)
for row in negTestData:
    testData.append(row)

results = []
sizePosTest = len(posTestData)
sizeNegTest = len(negTestData)
for i in range(sizePosTest):
    results.append(1)
for i in range(sizeNegTest):
    results.append(0)

#Naive Bayes Test
nBpositive, nBnegative = nB.test(posTestData, negTestData)
falsePositive = sizeNegTest - nBnegative
falseNegative = sizePosTest - nBpositive

# print test statistics
print("Naive bayes statistics for " + str(sizeNegTrain + sizePosTrain) + " train examples are the above:")
print("True Positive: " + str(nBpositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(nBnegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((nBpositive / sizePosTest) * 100), 2)))   # 84.58 
print("The accuracy for the negative data is: " + str(round(((nBnegative / sizeNegTest) * 100), 2)))  # 78.3
print("The total accuracy is: " + str(round((((nBnegative + nBpositive) / (sizeNegTest + sizePosTest)) * 100), 2)))  
precision = round((nBpositive / (nBpositive + falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((nBpositive / (nBpositive + falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2 / (1 / recall + 1 / precision), 3)))

#Random forest test
rfpositive, rfnegative = rf.test(testData)
falsePositive = sizeNegTest -rfnegative
falseNegative = sizePosTest - rfpositive

# print train statistics
print("Random Forest statistics for " + str(sizeNegTrain + sizePosTrain) + " train examples are the above:")
print("True Positive: " + str(rfpositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(rfnegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((rfpositive / sizePosTest)), 3)))
print("The accuracy for the negative data is: " + str(round(((rfnegative / sizePosTest)), 3)))
print("The total accuracy is: " + str(round((((rfnegative + rfpositive) / (sizeNegTest + sizePosTest))), 3)))  
precision = round((rfpositive / (rfpositive + falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((rfpositive / (rfpositive + falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2 / (1 / recall + 1 / precision), 3)))


res = Ada.predict(posTestData)
correct = 0
for i in res:
    if (i == 1):
        correct += 1

print("The accuracy for the positive test data was " + str(correct / len(posTestData)))

res = Ada.predict(negTestData)
correct = 0
for i in res:
    if (i == 0):
        correct += 1


print("The accuracy for the negative test data was " + str(correct / len(negTestData)))

