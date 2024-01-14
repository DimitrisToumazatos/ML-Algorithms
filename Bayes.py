from operator import add
from math import log2



class NaiveBayes:

  def __init__(self):
    self.probPos = 0
    self.probNeg = 0

  # In this function we calculate all
  # the probabilities of the existence of a feature in the
  # negative and the positive examples.
  def fit(self, trainData, trainResults):

    print("Training has started.")

    posData = []
    negData = []

    for i in range(len(trainResults)):                  # Split the train data into positive and negatives
      if trainResults[i] == 1:
        posData.append(trainData[i])
      else:
        negData.append(trainData[i])

    print("Calculating the probability for the positive training data...")

    numberOfPositive = len(posData)    
    pos_vocabulary_count = [0] * len(trainData[0])                                    # create a table of zeros with length of a row
    
    for row in posData:
      pos_vocabulary_count = list(map(add, pos_vocabulary_count, row))                # get the count of each feature in the positive examples
    
    self.pos_vocabulary_prob = [x / numberOfPositive for x in pos_vocabulary_count]   # calculate the probability of each feature for the positive examples 
    

    print("Calculating the probability for the negative training data...")

    numberOfNegative = len(negData)      
    neg_vocabulary_count = [0] * len(negData)                                         # create a table of zeros the length of a row

    for row  in negData:  
      neg_vocabulary_count = list(map(add, neg_vocabulary_count, row))                # get the count of each feature in the negative examples
    
    self.neg_vocabulary_prob = [x / numberOfNegative for x in neg_vocabulary_count]   # calculate the probability of each feature for the negative examples 

    self.probPos = numberOfPositive / (numberOfNegative + numberOfPositive)    # calculate the probability of an example being positive
    self.probNeg = numberOfNegative / (numberOfNegative + numberOfPositive)    # calculate the probability of an example being negative

  # In this function we calculate the 
  # probability each given test example being
  # classified as negative or positive 
  # and we return the predictions.
  def predict(self, testData):
    print("Testing has started.")

    prediction = []
    for row in testData:
      probP = self.probPos * self.calculateProb(1, row)    # Calculate the probability an example is positive depending on its vector
      probN = self.probNeg * self.calculateProb(0, row)    # Calculate the probability an example is negative depending on its vector

      if probP > probN:                                    # Make the prediction
        prediction.append(1)
      elif probP < probN:
        prediction.append(0)
      else:
        if self.probPos > self.probNeg:
          prediction.append(1)
        else:
          prediction.append(0)

    print("Test has finished.")
    return prediction

  # In this function we calculate the
  # classification probability. 
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

