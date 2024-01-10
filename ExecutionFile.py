from Bayes import *
from AdaBoost import *
import pandas as pd
import copy

PosTrainData = []
NegTrainData = []
#Reading of the Positive Training Data
print("Reading of the positive training data!")

dfPos = pd.read_csv("positiveTrain.csv", header = None)     # read the positive training examples
sizePos = dfPos.shape[0]        # shape[0] contains the number of rows of our file

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    PosTrainData.append(row)

#Reading of the Negative Training Data
print("Reading of the negative training data!")

dfNeg = pd.read_csv("negativeTrain.csv", header = None)     # read the negative training examples
sizeNeg = dfNeg.shape[0]        # shape[0] contains the number of rows of our file

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    NegTrainData.append(row)

 

PosTestData = []
NegTestData = []
#Reading of the Positive Test Data
print("Reading the positive test data!")

dfPos = pd.read_csv("positiveDev.csv", header = None)     # read the positive test examples
sizePos = dfPos.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]  # df.iloc function returns the row of our csv that we tell it to give. 
    PosTestData.append(row)

#Reading of the Negative Test Data
print("Reading of the negative test data!")

dfNeg = pd.read_csv("negativeDev.csv", header = None)     # read the negative test examples
sizeNeg = dfNeg.shape[0]                                  # shape[0] contains the number of rows of our file

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    NegTestData.append(row)


"""
myNaiveBayes = NaiveBayes()
myNaiveBayes.train(PosTrainData, dfPos.shape[1], NegTrainData, dfNeg.shape[1])

with open('myBayes-130-1700.pkl', 'wb') as outp:
    pickle.dump(myNaiveBayes, outp, pickle.HIGHEST_PROTOCOL)
"""
posTestData = []
negTestData = []
#Reading of the Positive Test Data
print("Reading of the positive Test Data!")

dfPos = pd.read_csv("positiveTest.csv", header = None)
sizePos = dfPos.shape[0]

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i, : ]]
    posTestData.append(row)

res = Ada.predict(posTestData)
correct = 0
for i in res:
    if (i == 1):
        correct += 1

print("The accuracy for the positive test data was " + str(correct / len(res)))

#Reading of the negative test Data  
print("Reading of the negative Test Data!")

dfNeg = pd.read_csv("negativeTest.csv", header = None)
sizeNeg = dfNeg.shape[0]

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i, : ]]
    negTestData.append(row)
"""

Train_Data = copy.deepcopy(PosTrainData)
for row in NegTrainData:
    Train_Data.append(row)
results = []
for i in range(len(PosTrainData)):
    results.append(1)
for i in range(len(NegTrainData)):
    results.append(0)

Ada = AdaBoost(Train_Data, results, 50)
Ada.fit()

print("Save model...")

with open('myAdaBoost2.pkl', 'wb') as outp:            # save object 
    pickle.dump(Ada, outp, pickle.HIGHEST_PROTOCOL) 
    """
with open('myAdaBoost.pkl', 'rb') as inp:             # read saved object
    Ada = pickle.load(inp)
"""


testData = copy.deepcopy(PosTestData)
for row in NegTestData:
    testData.append(row)

results = []
numberOfPositive = len(PosTestData)
numberOfNegative = len(NegTestData)
for i in range(numberOfPositive):
    results.append(1)
for i in range(numberOfNegative):
    results.append(0)



truePositive = 0
for i in range(numberOfPositive):
    prediction = Ada.predict(testData[i])
    #print("positive " + str(prediction))

    if prediction == results[i]:
        truePositive += 1

trueNegative = 0
for i in range(numberOfPositive, numberOfPositive+numberOfNegative):
    prediction = Ada.predict(testData[i])
    #print("negative " + str(prediction))
    if prediction == results[i]:
        trueNegative += 1




print("The accuracy on the Positive data is: " + str(truePositive*100/numberOfPositive))
print("The Accuracy on the Negative data is: " + str(trueNegative*100/numberOfNegative))


res = Ada.predict(negTestData)
correct = 0
for i in res:
    if (i == 0):
        correct += 1

print("The accuracy for the negative test data was " + str(correct / len(negTestData)))




"""
#Calling the Bayes algorithm 
print("Now we run the Bayes algorithm")

with open('myBayes-130-1700.pkl', 'rb') as inp:
    myNaiveBayes = pickle.load(inp)


myNaiveBayes.test(posTestData, negTestData)
