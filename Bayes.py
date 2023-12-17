import csv
from operator import add
import pandas as pd
from math import log2



class NaiveBayes:
  pos_vocabulary_prob = []
  neg_vocabulary_prob = []
  prob_pos = 0
  prob_neg = 0


  def __init__(self):
    pass

  def train(self, pos_filename, neg_filename):
    nNeg = 0
    df = pd.read_csv(pos_filename, header=None)
    nPos = df.shape[0]      #shape[0] contains the number of rows + 1 of our file
    
    pos_vocabulary_count = [0 for j in range(int(df.size / nPos))]  #it creates a table of zeros the length of a row

    for i in range( nPos):
      row = [int(x) for x in df.iloc[i, :]]   #df.iloc function returns the row of our csv that we tell it to give. Here we create a table with the elements of each row
      pos_vocabulary_count = list(map( add, pos_vocabulary_count, row))
    

    self.pos_vocabulary_prob = [x / nPos for x in pos_vocabulary_count]
    

    df = pd.read_csv(neg_filename, header = None)
    nNeg = df.shape[0]    
    
    neg_vocabulary_count = [0 for j in range(int((df.size / nNeg)))]   #it creates a table of zeros the length of a row
    for i in range(nNeg):
      row = [int(x) for x in df.iloc[i, :]]     #df.iloc function returns the row of our csv that we tell it to give. Here we create a table with the elements of each row
      neg_vocabulary_count = list(map( add, neg_vocabulary_count, row))

    self.prob_pos = nPos / (nNeg + nPos)
    self.prob_neg = nNeg / (nNeg + nPos)

    
    self.neg_vocabulary_prob = [x / nNeg for x in neg_vocabulary_count]

    with open('modelData.csv', 'w', newline='', encoding="utf8") as csv_file:        # save model
      writer = csv.writer(csv_file)                
      writer.writerow(self.pos_vocabulary_prob)
      writer.writerow(self.neg_vocabulary_prob)
      writer.writerow([self.prob_pos])
      writer.writerow([self.prob_neg])

  
  def test(self, pos_filename, neg_filename):
    countPos = 0
    df = pd.read_csv(pos_filename, header = None)
    for i in range(df.shape[0]):
      row = [int(x) for x in df.iloc[i, :]]
      probP = self.prob_pos * self.calculateProb(1, row)    #We calculate the probability an example is positive depending its produced vector
      probN = self.prob_neg * self.calculateProb(0, row)    #We calculate the probability an example is negative depending its produced vector
      if probP >= probN:
        countPos += 1

    print("For the positive examples the accuracy is: " + str(round(((countPos/df.shape[0]) * 100), 2)))   #86.56  


    countNeg = 0
    df = pd.read_csv(neg_filename, header = None)
    for i in range(df.shape[0]):
      row = [int(x) for x in df.iloc[i, :]]
      probP = self.prob_pos * self.calculateProb(1, row)
      probN = self.prob_neg * self.calculateProb(0, row)
        
      if probP <= probN:
        countNeg += 1

    
    print("For the negative examples the accuracy is: " + str(round(((countNeg/df.shape[0]) * 100), 2)))  #72.24
    

  def calculateProb(self, category, row):
    prob = 0
    if category == 1:
      l = self.pos_vocabulary_prob
    else:
      l = self.neg_vocabulary_prob

    index = 0
    for i in row:
      if i == 1:
        prob += log2(l[index] + 1)    #We use maximum likelihood using logarithm with base 2 because a probability can be literally 0 and we want a more accurate result
      else:
        prob += log2((1 - l[index]) + 1)

      index += 1

    return prob


Nb = NaiveBayes()
Nb.train("positive.csv", "negative.csv")
Nb.test("positiveDev.csv", "negativeDev.csv")