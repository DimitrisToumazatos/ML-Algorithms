import csv
from operator import add
import pickle
import pandas as pd
from math import log2



class NaiveBayes:

  def __init__(self):
    self.pos_vocabulary_prob = []
    self.neg_vocabulary_prob = []
    self.prob_pos = 0
    self.prob_neg = 0

  def train(self, pos_filename, neg_filename):

    print("Training has started.")

    print("Calculating the probability for the positive training data...")

    df = pd.read_csv(pos_filename, header=None)                     # read the positive training examples
    nPos = df.shape[0]                                              # shape[0] contains the number of rows of our file
    
    pos_vocabulary_count = [0 for j in range(df.shape[1])]          # it creates a table of zeros the length of a row
    
    for i in range(nPos):
      row = [int(x) for x in df.iloc[i, :]]                               # df.iloc function returns the row of our csv that we tell it to give. 
                                                                          # here we create a table with the elements of each row 
      pos_vocabulary_count = list(map( add, pos_vocabulary_count, row))   # get the count of each word (feature) in the negative examples
    
    self.pos_vocabulary_prob = [x / nPos for x in pos_vocabulary_count]   # calculate the probability of each feature for the positive examples 
    

    print("Calculating the probability for the negative training data...")

    df = pd.read_csv(neg_filename, header = None)                         # read the negative training examples
    nNeg = df.shape[0]    
    
    neg_vocabulary_count = [0 for j in range(df.shape[1])]                # create a table of zeros the length of a row

    for i in range(nNeg):     
      row = [int(x) for x in df.iloc[i, :]]                               # create a table with the elements of each row
      neg_vocabulary_count = list(map( add, neg_vocabulary_count, row))   # get the count of each word (feature) in the negative examples
    
    self.neg_vocabulary_prob = [x / nNeg for x in neg_vocabulary_count]   # calculate the probability of each feature for the negative examples 

    self.prob_pos = nPos / (nNeg + nPos)
    self.prob_neg = nNeg / (nNeg + nPos)


  def test(self, pos_filename, neg_filename):

    print("Testing has started.")
    

    print("Testing positive test data...")

    truePositive = 0
    df = pd.read_csv(pos_filename, header = None)           # read the positive test data

    numberOfPositive = df.shape[0]
    for i in range(numberOfPositive):
      row = [int(x) for x in df.iloc[i, :]]
      probP = self.prob_pos * self.calculateProb(1, row)    # calculate the probability an example is positive depending on its vector
      probN = self.prob_neg * self.calculateProb(0, row)    # calculate the probability an example is negative depending on its vector

      if probP >= probN:
        truePositive += 1


    print("Testing negative test data...")

    trueNegative = 0
    df = pd.read_csv(neg_filename, header = None)           # read the negative test data

    numberOfNegative = df.shape[0]
    for i in range(numberOfNegative):
      row = [int(x) for x in df.iloc[i, :]]
      probP = self.prob_pos * self.calculateProb(1, row)    # calculate the probability an example is positive depending on its vector
      probN = self.prob_neg * self.calculateProb(0, row)    # calculate the probability an example is negative depending on its vector
        
      if probP <= probN:
        trueNegative += 1

    falsePositive = numberOfNegative - trueNegative
    falseNegative = numberOfPositive - truePositive

    # print test statistics
    print("True Positive: " + str(truePositive))
    print("False Positive: " + str(falsePositive))
    print("True Negative: " + str(trueNegative))
    print("False Negative: " + str(falseNegative))
    print("The accuracy for the positive data is: " + str(round(((truePositive/numberOfPositive) * 100), 2)))   # 84.58 
    print("The accuracy for the negative data is: " + str(round(((trueNegative/numberOfNegative) * 100), 2)))  # 78.3
    print("The total accuracy is: " + str(round((((trueNegative + truePositive)/(numberOfNegative + numberOfPositive)) * 100), 2)))  
    print("For the positive data the precision is: " + str(round((truePositive/(truePositive+falsePositive) * 100), 2)))   
    print("For the positive data the recall is: " + str(round((truePositive/(truePositive+falseNegative)) * 100, 2)))   


  def calculateProb(self, category, row):
    prob = 0
    if category == 1:
      l = self.pos_vocabulary_prob
    else:
      l = self.neg_vocabulary_prob

    index = 0
    for i in row:
      if i == 1:
        prob += log2(l[index] + 1)    # We use maximum likelihood in order to get a more accurate result. Using logarithm with base 2 we avoid the
                                      # of the probability being very close to 0 and the computer rounding down this number to 0.
      else:
        prob += log2((1 - l[index]) + 1)

      index += 1

    return prob


myNaiveBayes = NaiveBayes()
myNaiveBayes.train("positiveTrain.csv", "negativeTrain.csv")

"""
with open('myBayes-16k.pkl', 'wb') as outp:            # save object 
    pickle.dump(myNaiveBayes, outp, pickle.HIGHEST_PROTOCOL) 

with open('myBayes-20k.pkl', 'rb') as inp:             # read saved object
    myNaiveBayes = pickle.load(inp)
"""

myNaiveBayes.test("positiveTest.csv", "negativeTest.csv")
