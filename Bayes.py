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
    flag = 0
    nPos = 0
    nNeg = 0
    pos_vocabulary_count = []
    neg_vocabulary_count = []
    with open(pos_filename, 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        row = [int(x) for x in row]
        nPos += 1
        if (flag == 0):
          flag = 1
          pos_vocabulary_count = [0 for i in range(len(row))]
          neg_vocabulary_count = [0 for i in range(len(row))]
        pos_vocabulary_count = list(map( add, pos_vocabulary_count, row))
    


    with open(neg_filename, 'r') as file:
      csvreader = csv.reader(file)
      
      for row in csvreader:
        row = [int(x) for x in row]
        nNeg += 1
        neg_vocabulary_count = list(map( add, neg_vocabulary_count, row))

    self.prob_pos = nPos / (nNeg + nPos)
    self.prob_neg = nNeg / (nNeg + nPos)

    self.pos_vocabulary_prob = [x / nPos for x in pos_vocabulary_count]
    self.neg_vocabulary_prob = [x / nNeg for x in neg_vocabulary_count]

    with open('modelData.csv', 'w', newline='', encoding="utf8") as csv_file:        # save model
      writer = csv.writer(csv_file)                
      writer.writerow(self.pos_vocabulary_prob)
      writer.writerow(self.neg_vocabulary_prob)
      writer.writerow([self.prob_pos])
      writer.writerow([self.prob_neg])

  def test(self, pos_filename, neg_filename):
    nPos = 0
    nNeg = 0
    countPos = 0
    countNeg = 0
    with open(pos_filename, 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        nPos += 1
        row = [int(x) for x in row]
        probP = self.prob_pos * self.calculateProb(1, row)
        probN = self.prob_neg * self.calculateProb(0, row)
        
        if probP >= probN:
          countPos += 1

    countNeg = 0
    with open(neg_filename, 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        nNeg += 1
        row = [int(x) for x in row]
        probP = self.prob_pos * self.calculateProb(1, row)
        probN = self.prob_neg * self.calculateProb(0, row)
        
        if probP <= probN:
          countNeg += 1
      

    print("For the positive examples the accuracy is: " + str(probP/nPos))  
    print("For the negetive examples the accuracy is: " + str(probN/nNeg))  

  def calculateProb(self, category, row):
    prob = 0
    if category == 1:
      l = self.pos_vocabulary_prob
    else:
      l = self.neg_vocabulary_prob

    index = 0
    for i in row:

      if i == 1:
        prob += log2(l[index] + 1)
      else:
        prob += log2((1 - l[index]) + 1)

      index += 1

    return prob


Nb = NaiveBayes()
Nb.train("positive.csv", "negative.csv")
Nb.test("positiveDev.csv", "negativeDev.csv")



