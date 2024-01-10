import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


k = 10000 # Number of examples from 1 category

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

trainData = copy.deepcopy(posTrainData[:k])
resultsTrain = [1]*k

trainData += negTrainData[:k]

resultsTrain += [0]*k


######### Read Test Data ############
        
posTestData = []
negTestData = []

#Reading of the Positive Test Data
print("Reading the positive test data!")

dfPos = pd.read_csv("positiveTest.csv", header = None)     # read the positive test examples
sizePos = dfPos.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    posTestData.append(row)

#Reading of the Negative Test Data
print("Reading the negative test data!")

dfNeg = pd.read_csv("negativeTest.csv", header = None)     # read the negative test examples
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


###############################################################################################
    
########################## Bayes ##################################################3
"""
X_train, X_test, y_train, y_test = trainData, testData, resultsTrain, resultsTest
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

###### Test Data Test ######

print("\nTest Data Results \n")

truePositive = (y_test[:sizePos] == y_pred[:sizePos]).sum()
trueNegative = (y_test[sizePos:] == y_pred[sizePos:]).sum()
falsePositive = sizeNeg - trueNegative
falseNegative = sizePos - truePositive

# print test statistics
print("True Positive: " + str(truePositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(trueNegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((truePositive/numberOfPositive)), 3)))
print("The accuracy for the negative data is: " + str(round(((trueNegative/numberOfNegative)), 3)))
print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(numberOfNegative + numberOfPositive))), 3)))  
precision = round((truePositive/(truePositive+falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((truePositive/(truePositive+falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  



##### Train Data Test #######
print("\nTrain Data Results \n")
X_train, X_test, y_train, y_test = trainData, trainData, resultsTrain, resultsTrain
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

truePositive = (y_test[:k] == y_pred[:k]).sum()
trueNegative = (y_test[k:] == y_pred[k:]).sum()
falsePositive = k - trueNegative
falseNegative = k - truePositive

# print train statistics
print("True Positive: " + str(truePositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(trueNegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((truePositive/k)), 3)))
print("The accuracy for the negative data is: " + str(round(((trueNegative/k)), 3)))
print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(k + k))), 3)))  
precision = round((truePositive/(truePositive+falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((truePositive/(truePositive+falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  

"""
###########################################################################################################

###########3 Random Forest #######################################################

X, y = trainData, resultsTrain
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
y_pred = clf.predict(testData)
y_test = resultsTest
###### Test Data Test ######

print("\nTest Data Results \n")

truePositive = (y_test[:sizePos] == y_pred[:sizePos]).sum()
trueNegative = (y_test[sizePos:] == y_pred[sizePos:]).sum()
falsePositive = sizeNeg - trueNegative
falseNegative = sizePos - truePositive

# print test statistics
print("True Positive: " + str(truePositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(trueNegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((truePositive/numberOfPositive)), 3)))
print("The accuracy for the negative data is: " + str(round(((trueNegative/numberOfNegative)), 3)))
print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(numberOfNegative + numberOfPositive))), 3)))  
precision = round((truePositive/(truePositive+falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((truePositive/(truePositive+falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  



##### Train Data Test #######
print("\nTrain Data Results \n")
X_train, X_test, y_train, y_test = trainData, trainData, resultsTrain, resultsTrain
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

truePositive = (y_test[:k] == y_pred[:k]).sum()
trueNegative = (y_test[k:] == y_pred[k:]).sum()
falsePositive = k - trueNegative
falseNegative = k - truePositive

# print train statistics
print("True Positive: " + str(truePositive))
print("False Positive: " + str(falsePositive))
print("True Negative: " + str(trueNegative))
print("False Negative: " + str(falseNegative))
print("The accuracy for the positive data is: " + str(round(((truePositive/k)), 3)))
print("The accuracy for the negative data is: " + str(round(((trueNegative/k)), 3)))
print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(k + k))), 3)))  
precision = round((truePositive/(truePositive+falsePositive)), 3)
print("For the positive data the precision is: " + str(precision))  
recall =  round((truePositive/(truePositive+falseNegative)), 3)
print("For the positive data the recall is: " + str(recall))    
print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  

