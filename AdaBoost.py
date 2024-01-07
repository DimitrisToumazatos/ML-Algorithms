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
    
    def fit(self, weights, x, y):
        key = self.bestKey(x, y, weights)
        return 0

    def TableCreation(self, weights, x_train, y_train): 
        #Here we give the tree our train data
        #and we run th decision tree to rank the comments
        #to positive or negative depending on the value of the 
        #selected feature
        indices_0 = [x for x in range(len(x_train)) if x_train[x][self.feature] == 0]
        indices_1 = [x for x in range(len(x_train)) if x_train[x][self.feature] == 1]
        results_0 = len([x for x in indices_0 if y_train[x] == 0])
        if (results_0 > len(y_train) - results_0):
            category_0 = 0
        else:
            category_0 = 1
        results_1 = len([x for x in indices_1 if y_train[x] == 0])
        if (results_1 > len(y_train) - results_1):
            category_1 = 0
        else:
            category_1 = 1
        return category_0, category_1
    
    def bestKey(self, x_train, y_train, weights):
        keys = [i for i in range(1570)]
        gains = []
        allExamples = len(x_train)

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
                        count_1_negative += weights[j]
            
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

            gains.append(hc - (pc1 * hcX1) - ((1-pc1) * hcX0))

        maxgain = max(gains)
        return keys[gains.index(maxgain)]

    def binEntropy(self, prob):
        if (prob == 0 or prob == 1):
            return 0
        else:
            return - (prob * math.log2(prob)) - ((1-prob)*math.log2(1-prob))


    def predict(self, X_train):
        results = []
        for row in X_train:
            if (row[self.feature] == 0):
                results.append(self.category_0)
            else:
                results.append(self.category_1)
        return results
        
class AdaBoost:
    def __init__(self, Dataset_x, base_learner, M):
        self.Dataset = Dataset_x
        self.base_learner = base_learner
        self.m = M

    def fit(self, x_train, y_train): 
        self.W = [1 / len(self.Dataset) for _ in range(len(self.Dataset))]
        self.h_t = []
        self.z = []

        for i in range(self.m - 1):
            
            ht = self.base_learner.fit(maxIG, x_train, y_train, self.W)
            self.h_t.append(ht)

            algorithm_results = ht.predict(x_train)
            incorrect = 1 - (algorithm_results == y_train)
            print(incorrect)
            sleep(5)
            error = np.sum(self.W * incorrect) / np.sum(self.W, axis = 0)
            if (round(error, 2) > 0.51):
                self.m = _ - 1
                break
            elif (error <= 0.001):
                self.m = _
                break
            if (error != 0 and error != 1):
                self.z.append(0.5 * log2((1 - error)/ error))
            elif error == 0:
                self.z.append(1)
            for j in range(len(y_train)): #change weights based on the errors
                    if (algorithm_results[j] == y_train[j]):
                        if(error != 1):
                            self.weights[j] *= exp(-self.z[_])
                    elif (algorithm_results[j] != y_train[j]):
                        if (error != 1):
                            self.weights[j] *= exp(self.z[_])
            self.normalizeWeights()

    def MakeHypothesis(self):
            
            self.features.remove(self.features[maxIG])
            SDT = SingleDepthTree(maxIG)
            self.Hypotheses.append(SDT)
            print("Hypothesis made!")
            SDT.fit(self.weights, self.Dataset, self.expectedResults)
            algorithm_results = []
            for row in self.Dataset:
                algorithm_results.append(SDT.predict(row))
            return algorithm_results
            
    
    def normalizeWeights(self): #function to normalize weights to sum up to 1
        sum_1 = sum(self.weights)
        self.weights = [x / sum_1 for x in self.weights]

    def InformationGain(self, feature): #given a feature calculate the information gain
        classes = [0, 1]
        HC = 0
        
        for i in classes:
            PC = self.expectedResults.count(i) / len(self.expectedResults)
            HC -= PC * math.log2(PC)
        
        feature_values = [0, 1]
        HC_feature = 0

        for value in feature_values:
            pf = feature.count(value) / len(feature)  #the probability the value of our feature equals x(where x is all its possible values)
            indices = [ind for ind in range(len(feature)) if feature[i] == value]
            pf_weights = sum([self.weights[i] for i in indices])
            pf *= pf_weights
            for j in classes:
                pcf = self.Conditional_Probability_Calculation(feature, j, value)
                if (pcf != 0):
                    temp_H = -pf * pcf * math.log2(pcf)
                    HC_feature += temp_H

        ig = HC - HC_feature
        return ig         

    def Error_Calculation(self, Algorithm_results):
        error = 0
        for i in range(len(self.expectedResults)):
            if (Algorithm_results[i] != self.expectedResults[i]):
                error += self.weights[i]
        return error
    
    def Conditional_Probability_Calculation(self, features_table, class_value, feature_value):
        indices = [i for i in range(len(features_table)) if features_table[i] == feature_value]
        class_count = len([i for i in indices if self.expectedResults[i] == class_value])
        weights_count = sum([self.weights[i] for i in indices if self.expectedResults[i] == class_value])
        return (class_count * weights_count) / len(indices)

    def predict(self, TestData):
        sum_0 = 0
        sum_1 = 0
        for index in range(len(self.z)):
            prediction = self.Hypotheses[index].predict(TestData)
            if (prediction == 0):
                sum_0 += self.z[index]
            else:
                sum_1 += self.z[index]
        if (sum_0 > sum_1):
            return 0
        elif (sum_1 > sum_0):
            return 1
        else:
            randomNum = round(random.random())
            if (randomNum == 0):
                return 0
            else:
                return 1
    
    
