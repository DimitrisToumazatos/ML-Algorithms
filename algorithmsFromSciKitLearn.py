import copy
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

list1 = [2000, 4000, 6000, 8000, 10000] # Number of examples from 1 category

######### Read Test Data ############
            
posTestData = []
negTestData = []

#Reading of the Positive Test Data
print("Reading the positive test data!")

dfPos = pd.read_csv("positiveTest.csv", header = None)     # read the positive test examples
sizePosTest = dfPos.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizePosTest):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    posTestData.append(row)

#Reading of the Negative Test Data
print("Reading the negative test data!")

dfNeg = pd.read_csv("negativeTest.csv", header = None)     # read the negative test examples
sizeNegTest = dfNeg.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizeNegTest):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    negTestData.append(row)


testData = copy.deepcopy(posTestData)
for row in negTestData:
    testData.append(row)

y_test = []

for i in range(sizePosTest):
    y_test.append(1)
for i in range(sizeNegTest):
    y_test.append(0)

for k in list1:

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
    print("Reading the negative training data!")

    dfNeg = pd.read_csv("negativeTrain.csv", header = None)     # read the negative training examples
    sizeNeg = dfNeg.shape[0]        # shape[0] contains the number of rows of our file

    for i in range(sizeNeg):
        row = [int(x) for x in dfNeg.iloc[i, : ]]
        negTrainData.append(row)

    trainData = copy.deepcopy(posTrainData[:k])
    resultsTrain = [1]*k

    trainData += negTrainData[:k]

    resultsTrain += [0]*k
    
    ###############################################################################################
        
    ########################## Bayes ##################################################3
    

    gnb = GaussianNB()
    y_pred = gnb.fit(trainData, resultsTrain).predict(testData)

    ###### Test Data Test ######
    print("\nNaive Bayes for "+str(k+k)+ "training examples.\n")
    print("Statistics on Test Data:  \n")

    truePositive = 0
    trueNegative = 0
    for i in range(sizePosTest):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(sizePosTest, sizeNegTest+sizePosTest):
        if y_pred[i] == 0:
            trueNegative += 1

    falsePositive = sizeNegTest - trueNegative
    falseNegative = sizePosTest - truePositive

    # print test statistics
    print("True Positive: " + str(truePositive))
    print("False Positive: " + str(falsePositive))
    print("True Negative: " + str(trueNegative))
    print("False Negative: " + str(falseNegative))
    print("The accuracy for the positive data is: " + str(round(((truePositive/sizePosTest)), 3)))
    print("The accuracy for the negative data is: " + str(round(((trueNegative/sizeNegTest)), 3)))
    print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(sizeNegTest + sizePosTest))), 3)))  
    precision = round((truePositive/(truePositive+falsePositive)), 3)
    print("For the positive data the precision is: " + str(precision))  
    recall =  round((truePositive/(truePositive+falseNegative)), 3)
    print("For the positive data the recall is: " + str(recall))    
    print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  



    ##### Train Data Test #######
    print("\nStatistics on Train Data:  \n")

    y_pred = gnb.predict(trainData)

    truePositive = 0
    trueNegative = 0
    for i in range(k):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(k, k + k):
        if y_pred[i] == 0:
            trueNegative += 1

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



    ###########################################################################################################

    ############ Random Forest #######################################################

    X, y = trainData, resultsTrain
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    y_pred = clf.predict(testData)

    ###### Test Data Test ######

    print("\nRandom Forest for "+str(k+k)+ "training examples.\n")
    print("Statistics on Test Data:  \n")

    truePositive = 0
    trueNegative = 0
    for i in range(sizePosTest):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(sizePosTest, sizeNegTest+sizePosTest):
        if y_pred[i] == 0:
            trueNegative += 1


    falsePositive = sizeNegTest - trueNegative
    falseNegative = sizePosTest - truePositive

    # print test statistics
    print("True Positive: " + str(truePositive))
    print("False Positive: " + str(falsePositive))
    print("True Negative: " + str(trueNegative))
    print("False Negative: " + str(falseNegative))
    print("The accuracy for the positive data is: " + str(round(((truePositive/sizePosTest)), 3)))
    print("The accuracy for the negative data is: " + str(round(((trueNegative/sizeNegTest)), 3)))
    print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(sizeNegTest + sizePosTest))), 3)))  
    precision = round((truePositive/(truePositive+falsePositive)), 3)
    print("For the positive data the precision is: " + str(precision))  
    recall =  round((truePositive/(truePositive+falseNegative)), 3)
    print("For the positive data the recall is: " + str(recall))    
    print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))  



    ##### Train Data Test #######
    print("\nStatistics on Train Data:  \n")

    y_pred = clf.predict(trainData)


    truePositive = 0
    trueNegative = 0
    for i in range(k):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(k, k + k):
        if y_pred[i] == 0:
            trueNegative += 1



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
    


    
    ###################### AdaBoost ##########################################

    base_classifier = DecisionTreeClassifier(max_depth=1)
    adaBoost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, learning_rate=1.0)
    adaBoost_classifier.fit(trainData, resultsTrain)

    ###### Test Data Test ######
        
    print("\nAdaBoost for "+str(k+k)+ "training examples.\n")
    print("Statistics on Test Data:  \n")

    y_pred = adaBoost_classifier.predict(testData)


    truePositive = 0
    trueNegative = 0
    for i in range(sizePosTest):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(sizePosTest, sizeNegTest+sizePosTest):
        if y_pred[i] == 0:
            trueNegative += 1

    falsePositive = sizeNegTest - trueNegative
    falseNegative = sizePosTest - truePositive

    # print test statistics
    print("True Positive: " + str(truePositive))
    print("False Positive: " + str(falsePositive))
    print("True Negative: " + str(trueNegative))
    print("False Negative: " + str(falseNegative))
    print("The accuracy for the positive data is: " + str(round(((truePositive/sizePosTest)), 3)))
    print("The accuracy for the negative data is: " + str(round(((trueNegative/sizeNegTest)), 3)))
    print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(sizeNegTest + sizePosTest))), 3)))  
    precision = round((truePositive/(truePositive+falsePositive)), 3)
    print("For the positive data the precision is: " + str(precision))  
    recall =  round((truePositive/(truePositive+falseNegative)), 3)
    print("For the positive data the recall is: " + str(recall))    
    print("The F1 for the negative test data is: " + str(round(2/(1/recall + 1/precision), 3)))



    ##### Train Data Test #######
    print("\nStatistics on Train Data:  \n")

    y_pred = adaBoost_classifier.predict(trainData)

    truePositive = 0
    trueNegative = 0
    for i in range(k):
        if y_pred[i] == 1:
            truePositive += 1

    for i in range(k, k + k):
        if y_pred[i] == 0:
            trueNegative += 1

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
    
