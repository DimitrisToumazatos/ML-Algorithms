from cmath import log
import math
from random import random

class SingleDepthTree:  #We used the id3 algorithm given from our instructors adapted to create just single depth trees
    def __init__(self, features, category):
        self.category = category
        self.features = features
    
    def fit(self, x):
        self.table = self.TableCreation(x)

    def TableCreation(self, x_train): #Works a tree with depth = 1, the category is saved inside the table 
        table = []
        for i in x_train:
            if(i[self.feature] == 1):
                table.append(self.category)
            else:
                table.append(abs(1-self.category))
        return table
        
    
    
    
class AdaBoost:
    def __init__(self, Dataset_x, expectedResults, algorithm, M): # M = number of hypotheses
        self.Dataset = Dataset_x
        self.expectedResults = expectedResults
        self.algorithm = algorithm
        self.m = M
        self.Hypotheses = []
        self.weights = [1 / len(expectedResults) for i in range(len(expectedResults))] 
    
    
    def MakeHypotheses(self):
        for i in range(self.m):
            igs = []
            for feat_index in features:
                igs.append(self.InformationGain([example[feat_index] for example in self.Dataset]))
            maxIG = igs.index(max(igs))
            self.Hypotheses = self.algorithm(maxIG)
            error = 0
            for j in range(len(self.expectedResults)):
                if (self.Hypotheses[j] != self.expectedResults[j]): #if the hypothesis differs from the expected result, increase the error
                    error+=self.weights[j]
                if (error>=0.5):
                    self.m-=1
                    break
            for i in range(self.expectedResults): #change weights based on the errors
                    if (self.Hypotheses[j] == self.expectedResults[j]):
                        if(error != 1):
                            self.weights[i] *= error / (1-error)
            self.normalizeWeights()
            z = [] 
            for i in range(self.m):
                z[i].append(0.5*log(1-error/error))
            
            self.DatasetReconstruction() #increase or decrease the weights of each hypothesis based on if the hypothesis was correct
    
    def normalizeWeights(self): #function to normalize weights to sum up to 1
        sum = sum(self.weights)
        for weight in self.weights:
            weight = weight / sum
        return
    
    def DatasetReconstruction(self): #differentiation of the training examples based on their weights
        NDataset = []
        NWeights = []
        for i in range(len(self.weights)):
            rnum = random.random()
            sum = 0
            ind = 0
            for j in self.weights:
                sum += j
                if (sum >= rnum):
                    NDataset.append(self.Dataset[ind])
                    NWeights.append(self.weights[ind])
                    break
                ind += 1
        self.weights = NWeights
        self.Dataset = NDataset

    def InformationGain(self, feature): #given a feature calcuate the information gain
        classes = [0, 1]
        HC = 0
        
        for i in classes:
            PC = self.expectedResults.count(i) / len(self.expectedResults)
            HC -= PC * math.log(PC, 2)
        
        feature_values = [0, 1]
        HC_feature = 0

        for value in feature_values:
            pf = feature.count(value) / len(feature)
            indices = [i for i in range(len(feature)) if feature[i] == value]

            classes_of_feat = [self.expectedResults[i] for i in indices]
            for j in classes:
                pcf = classes_of_feat.count(j) / len(classes_of_feat)
                if (pcf != 0):
                    temp_H = -pf * pcf * math.log(pcf, 2)
                    HC_feature += temp_H

        ig = HC - HC_feature
        return ig

    