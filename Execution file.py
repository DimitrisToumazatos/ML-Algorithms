import Bayes, randomForest, AdaBoost
import pandas as pd

trainData = []
#Reading of the Positive Training Data
print("Reading the positive training data!")

dfPos = pd.read_csv("positive.csv")
sizePos = dfPos.shape[0]

for i in range(sizePos):
    row = [int(x) for x in dfPos.iloc[i: ]]
    trainData.append(row)

#Reading of the Negative Training Data
print("Reading of the negative training data")

dfNeg = pd.read_csv("negative.csv")
sizeNeg = dfNeg.shape[0]

for i in range(sizeNeg):
    row = [int(x) for x in dfNeg.iloc[i: ]]
    trainData.append(row)

