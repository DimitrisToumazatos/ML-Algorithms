from operator import add
from math import log2



class NaiveBayes:

  def __init__(self):
    self.prob_pos = 0
    self.prob_neg = 0

  def train(self, pos_Data, Pos_Columns, Neg_Data, Neg_Columns):
    #In the train function of naive Bayes we calculate all
    #the probabilities of the existence of a feature in the
    #negative and the positive examples respectively

    print("Training has started.")

    print("Calculating the probability for the positive training data...")

    nPos = len(pos_Data)
    
    pos_vocabulary_count = [0 for j in range(Pos_Columns)]          # it creates a table of zeros with length of a row
    
    for row in pos_Data:
      pos_vocabulary_count = list(map( add, pos_vocabulary_count, row))   # get the count of each word (feature) in the negative examples
    
    self.pos_vocabulary_prob = [x / nPos for x in pos_vocabulary_count]   # calculate the probability of each feature for the positive examples 
    

    print("Calculating the probability for the negative training data...")

    nNeg = len(Neg_Data)  
    
    neg_vocabulary_count = [0 for j in range(Neg_Columns)]                # create a table of zeros the length of a row

    for row  in Neg_Data:  
      neg_vocabulary_count = list(map(add, neg_vocabulary_count, row))   # get the count of each word (feature) in the negative examples
    
    self.neg_vocabulary_prob = [x / nNeg for x in neg_vocabulary_count]   # calculate the probability of each feature for the negative examples 

    self.prob_pos = nPos / (nNeg + nPos)    #calculation of the probability an example is positive
    self.prob_neg = nNeg / (nNeg + nPos)    #calculation of the probability an example is negative

  def test(self, pos_Data, neg_Data):
    #We calculate the probability each row(review)
    #is classified in the negatives or positives 
    #and we return the number of them respectively
    print("Testing has started.")

    print("Testing positive test data...")

    bayesPositive = 0
    
    for row in pos_Data:
      probP = self.prob_pos * self.calculateProb(1, row)    # calculate the probability an example is positive depending on its vector
      probN = self.prob_neg * self.calculateProb(0, row)    # calculate the probability an example is negative depending on its vector

      if probP >= probN:
        bayesPositive += 1


    print("Testing negative test data...")

    bayesNegative = 0
    
    for row in neg_Data:
      probP = self.prob_pos * self.calculateProb(1, row)    # calculate the probability an example is positive depending on its vector
      probN = self.prob_neg * self.calculateProb(0, row)    # calculate the probability an example is negative depending on its vector
        
      if probP <= probN:
        bayesNegative += 1

    return bayesPositive, bayesNegative


  def calculateProb(self, category, row):
    #In this function we do the calculation
    #of the classification probability for each 
    #row(review)
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

