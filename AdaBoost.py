import random
from time import sleep
import pandas as pd
import numpy as np        
from math import exp, log2
import math

class SingleDepthTree:  
    #This class creates a tree with just 2 leafs
    #based on the feature with the biggest information gain 
    #and makes a guess if it belongs to positive or negative
    def __init__(self):
        #The table contains the most possible class
        #an example belongs based the value of one of 
        # their features
        self.category_0 = -1
        self.category_1 = -1
        self.feature = -1
        
    
    def fit(self, weights, x_train, y_train, keys):
        self.feature = self.bestKey(x_train, y_train, weights, keys)
        count_1_positive = 0
        count_0_positive = 0
        count_1_negative = 0
        count_0_negative = 0

        for j in range(len(x_train)):
            if(y_train[j] == 1):
                if(x_train[j][self.feature] == 1):
                    count_1_positive += weights[j]
                else:
                    count_1_negative += weights[j]
            else:
                if(x_train[j][self.feature] == 1):
                    count_0_positive += weights[j]
                else:
                    count_0_negative += weights[j]
        
        #prob (C = 1 | X = 0)
        pC1X0 = 0
        if (count_1_positive + count_1_negative != 0):
            pC1X0 = float(count_1_positive / (count_1_positive + count_1_negative))
        #prob (C = 1 | X = 1)
        pC1X1 = 0
        if (count_1_positive + count_1_negative != 0):
            pC1X1 = float(count_0_positive / (count_0_positive + count_0_negative))
        if (pC1X0 > (1 - pC1X0)):
            self.category_0 = 1
        else:
            self.category_0 = 0
        self.category_1 = abs(1 - self.category_0)
    
    def bestKey(self, x_train, y_train, weights, keys):
        gains = []

        for i in keys:
            count_1_positive = 0
            count_0_positive = 0
            count_1_negative = 0
            count_0_negative = 0

            for j in range(len(x_train)):
                if(y_train[j] == 1):
                    if(x_train[j][i] == 1):
                        count_1_positive += weights[j]
                    else:
                        count_1_negative += weights[j]
                else:
                    if(x_train[j][i] == 1):
                        count_0_positive += weights[j]
                    else:
                        count_0_negative += weights[j]
            
            #prob C = 1
            pC1 = (count_1_positive + count_0_positive) / sum(weights)
            #prob (C = 1 | X = 0)
            pC1X0 = 0
            if (count_1_positive + count_1_negative != 0):
                pC1X0 = float(count_1_positive / (count_1_positive + count_1_negative))
            #prob (C = 1 | X = 1)
            pC1X1 = 0
            if (count_1_positive + count_1_negative != 0):
                pC1X1 = float(count_0_positive / (count_0_positive + count_0_negative))
            #entropies
            hcX1 = self.binEntropy(pC1X1)
            hcX0 = self.binEntropy(pC1X0)

            #Calculate initial binary entropy
            count_pos = count_0_positive + count_1_positive
            hc = self.binEntropy(count_pos / len(x_train))

            gains.append(hc - (pC1 * hcX1) - ((1-pC1) * hcX0))

        maxgain = max(gains)
        return keys[gains.index(maxgain)]

    def binEntropy(self, prob):
        if (prob == 0 or prob == 1):
            return 0
        else:
            return - (prob * math.log2(prob)) - ((1-prob)*math.log2(1-prob))

    def predict_row(self, row):
        if (row[self.feature] == 0):
            return self.category_0
        else:
            return self.category_1

    def predict(self, X_train):
        results = []
        for row in X_train:
            if (row[self.feature] == 0):
                results.append(self.category_0)
            else:
                results.append(self.category_1)
        return results
        
class AdaBoost:
    def __init__(self, Dataset_x, M):
        self.Dataset = Dataset_x
        self.m = M

    def fit(self, x_train, y_train): 
        self.W = [1 / len(self.Dataset) for _ in range(len(self.Dataset))]
        self.h_t = []
        self.z = []
        self.keys = [i for i in range(1570)]
        self.keys.remove(637)
        self.keys.remove(742)
        self.keys.remove(419)
        self.keys.remove(268)
        self.keys.remove(598)
        self.keys.remove(703)
        self.keys.remove(361)
        for i in range(self.m - 1):
            ht = SingleDepthTree()
            ht.fit(self.W, x_train, y_train, self.keys)
            print(ht.feature)
            self.h_t.append(ht)
            self.keys.remove(ht.feature)
            algorithm_results = ht.predict(x_train)
            error = 0
            for j in range(len(y_train)):
                if(y_train[j] != algorithm_results[j]):
                    error += self.W[j]
            print(error)
            if (round(error, 2) > 0.51):
                self.m = i - 1
                break
            elif (error <= 0.001):
                self.m = i
                break
            if (error != 0 and error != 1):
                self.z.append(0.5 * log2((1 - error)/ error))
            elif error == 0:
                self.z.append(1)
            for j in range(len(y_train)): #change weights based on the errors
                    if (algorithm_results[j] == y_train[j]):
                        if(error != 1):
                            self.W[j] *= (error / (1 - error))
            self.normalizeWeights()

    def normalizeWeights(self): #function to normalize weights to sum up to 1
        sum_1 = sum(self.W)
        self.W = [x / sum_1 for x in self.W]

    def predict(self, TestData):
        results = []
        print(len(self.h_t))
        for row in TestData:
            sum_0 = 0
            sum_1 = 0
            index = 0
            for h in self.h_t:
                prediction = h.predict_row(row)
                if (prediction == 0):
                    sum_0 += self.z[index]
                else:
                    sum_1 += self.z[index]
                index += 1
                if(index == len(self.z)):
                    break
            if (sum_0 > sum_1):
                results.append(0)
            elif (sum_1 > sum_0):
                results.append(1)
            else:
                randomNum = round(random.random())
                if (randomNum == 0):
                    results.append(0)
                else:
                    results.append(1)
        return results
    
    
