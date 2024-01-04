from math import log2
import math
import random
import copy

class SingleDepthTree:  #We used the id3 algorithm given from our instructors adapted to create just single depth trees
    def __init__(self, feature):
        self.feature = feature
        self.table = []
    
    def fit(self, x, y):
        category_0, category_1 = self.TableCreation(x, y)
        self.table.append(category_0)
        self.table.append(category_1)

    def TableCreation(self, x_train, y_train): #Works a tree with depth = 1, the category is saved inside the table 
        results_0 = []
        results_1 = []
        category_1 = -1
        category_0 = -1
        for i in range(len(x_train)):
            if(x_train[i][self.feature] == 1):
                results_1.append(y_train[i])
            else:
                results_0.append(y_train[i])
        if (results_1.count(0) > results_1.count(1)):
            category_1 = 0
        else:
            category_1 = 1
        if (results_0.count(0) > results_0.count(1)):
            category_0 = 0
        else:
            category_0 = 1
        return category_0, category_1

    def predict(self, row):
        if (row[self.feature] == 0):
            return self.table[0]
        else:
            return self.table[1]
        
    
    
    
class AdaBoost:
    def __init__(self, Dataset_x, expectedResults, M): # M = number of hypotheses
        self.Dataset = Dataset_x
        self.expectedResults = expectedResults
        self.m = M
        self.z = []
        self.Hypotheses = []
        self.weights = [1 / len(expectedResults) for i in range(len(expectedResults))]

    def fit(self):
        features = []
        for i in range(self.m):
            features.append(random.randint(0, 1570))
        self.MakeHypotheses(features)
    
    def MakeHypotheses(self, features):
        for i in range(self.m):   #We use this for loop to create our Hypotheses
            igs = []            #Here we calculate the igs of all the features 
            for feat_index in features:
                igs.append(self.InformationGain([example[feat_index] for example in self.Dataset]))
            maxIG = igs.index(max(igs))
            SDT = SingleDepthTree(maxIG)
            self.Hypotheses.append(SDT)
            SDT.fit(self.Dataset, self.expectedResults)
            algorithm_results = []
            for row in self.Dataset:
                algorithm_results.append(SDT.predict(row))
            error = 0
            
            for j in range(len(self.expectedResults)):
                if (algorithm_results[j] != self.expectedResults[j]): #if the hypothesis differs from the expected result, increase the error
                    error += self.weights[j]
            print(error)
            if (error >= 0.5):
                self.m -= 1
                break
            for j in range(len(self.expectedResults)): #change weights based on the errors
                    if (algorithm_results[j] == self.expectedResults[j]):
                        if(error != 1):
                            self.weights[j] *= error / (1-error)
            self.normalizeWeights()
            for i in range(self.m):
                if (error != 0 and error != 1):
                    self.z.append(0.5 * log2((1 - error)/ error))
                elif error == 0:
                    self.z.append(1)
                else:
                    self.z.append(0)
            print(len(self.Dataset))
            self.DatasetReconstruction() #Reconstructs the dataset based on the weight of each example given
            print(len(self.Dataset))
        self.Weighted_Majority()
    
    def normalizeWeights(self): #function to normalize weights to sum up to 1
        sum_1 = sum(self.weights)
        for weight in self.weights:
            weight = weight / sum_1
    
    def DatasetReconstruction(self): #differentiation of the training examples based on their weights     
        NDataset = []
        NWeights = []
        NExpected_Results = []
        for i in range(len(self.weights)):
            rnum = random.random()
            sum_1 = 0
            ind = 0
            for j in self.weights:
                if (sum_1 >= rnum):
                    NDataset.append(self.Dataset[ind])
                    NWeights.append(self.weights[ind])
                    NExpected_Results.append(self.expectedResults[ind])
                    break
                sum_1 += j
                ind += 1
            if (sum_1 == 1):
                NDataset.append(self.Dataset[-1])
                NWeights.append(self.weights[-1])
                NExpected_Results.append(self.expectedResults[-1])
        self.weights = NWeights
        self.Dataset = NDataset

    def InformationGain(self, feature): #given a feature calculate the information gain
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
    
    def Weighted_Majority(self):
        Zeros_weight = 0
        Ones_weight = 0
        for i in range(self.m):
            if (self.Hypotheses[i] == 0):
                Zeros_weight += self.z[i]
            else:
                Ones_weight += self.z[i]
        if (Zeros_weight > Ones_weight):
            return 0
        elif (Ones_weight > Zeros_weight):
            return 1
        else:
            randomNum = round(random.random())
            if (randomNum == 0):
                return 0
            else:
                return 1
            
    
    def predict(self, TestData):
        sum_0 = 0
        sum_1 = 0
        ind = 0
        for tree in self.Hypotheses:
            prediction = tree.predict(TestData)
            if (prediction == 0):
                sum_0 += self.z[ind]
            else:
                sum_1 += self.z[ind]
            ind += 1
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
