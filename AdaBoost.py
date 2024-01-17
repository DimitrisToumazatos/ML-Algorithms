import random
from math import exp, log2

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
        #We use the feature with the less gini index to create the tree
        #and we calculate the p(C = 1 | X = 0) and the p(C = 1 | X = 1) 
        self.feature = self.giniIndex(x_train, y_train, weights, keys)
        count_1_positive = 0
        count_0_positive = 0
        count_1_negative = 0
        count_0_negative = 0

        #the calculation of all the counters
        for j in range(len(x_train)):
            if(y_train[j] == 1):
                if(x_train[j][self.feature] == 1):
                    count_1_positive += weights[j]
                else:
                    count_0_positive += weights[j]
            else:
                if(x_train[j][self.feature] == 0):
                    count_0_negative += weights[j]
                else:
                    count_1_negative += weights[j]
        
        #prob (C = 1 | X = 0)
        pC1X0 = 0
        if (count_0_positive + count_0_negative != 0):
            pC1X0 = float(count_0_positive / (count_0_positive + count_0_negative))

        #prob (C = 1 | X = 1)
        pC1X1 = 0
        if (count_1_positive + count_1_negative != 0):
            pC1X1 = float(count_1_positive / (count_1_positive + count_1_negative))

        #prob (C = 0 | X = 0)
        pC0X0 = 0
        if (count_0_positive + count_0_negative != 0):
            pC0X0 = float(count_0_negative / (count_0_positive + count_0_negative))

        #prob (C = 0 | X = 1)
        pC0X1 = 0
        if (count_1_positive + count_1_negative != 0):
            pC0X1 = float(count_1_negative / (count_1_positive + count_1_negative))
        
        if (pC0X0 > pC1X0):
            self.category_0 = 0
        else:
            self.category_0 = 1

        if (pC0X1 > pC1X1):
            self.category_1 = 0
        else:
            self.category_1 = 1
        

    def giniIndex(self, x_train, y_train, weights, keys):
        #Gini index uses the squares sum of the p(C = 1 | X = 0) 
        #and the p(C = 1 | X = 1) and their inverse 
        GiniIndex = []

        for i in keys:
            count_1_positive = 0
            count_0_positive = 0
            count_1_negative = 0
            count_0_negative = 0

            #the calculation of all the counters
            for j in range(len(x_train)):
                if(y_train[j] == 1):
                    if(x_train[j][i] == 1):
                        count_1_positive += weights[j]
                    else:
                        count_0_positive += weights[j]
                else:
                    if(x_train[j][i] == 0):
                        count_0_negative += weights[j]
                    else:
                        count_1_negative += weights[j]
            
            #prob (C = 1 | X = 0)
            pC1X0 = 0
            if (count_0_positive + count_0_negative != 0):
                pC1X0 = float(count_0_positive / (count_0_positive + count_0_negative))
            #prob (C = 1 | X = 1)
            pC1X1 = 0
            if (count_1_positive + count_1_negative != 0):
                pC1X1 = float(count_1_positive / (count_1_positive + count_1_negative))
            giniIndex1 = 1 - (pC1X1 ** 2 + (1 - pC1X1) ** 2)
            giniIndex0 = 1 - (pC1X0 ** 2 + (1 - pC1X0) ** 2)
            giniIndex = ((count_0_negative + count_0_positive) * giniIndex0) + ((count_1_negative + count_1_positive) * giniIndex1)

            GiniIndex.append(giniIndex)
        
        minGini = min(GiniIndex)
        return keys[GiniIndex.index(minGini)]
            
    
    def predict_row(self, row):
        #uses the value of the selected feature
        #of the row to predict its classification 
        if (row[self.feature] == 0):
            return self.category_0
        else:
            return self.category_1

    def predict(self, testData):
        #uses the value of the selected feature
        #of the testData to predict their classification
        results = []
        for row in testData:
            if (row[self.feature] == 0):
                results.append(self.category_0)
            else:
                results.append(self.category_1)
        return results
        
class AdaBoost:
    def __init__(self, M):
        #sets the m feature which
        #defines the number of trees created
        self.m = M


        #initialize all the tables we gonna need
        #and create all the trees with the best feature each time,
        #calculate their error, update their weights and set each
        #hypothesis weight for the prediction
    def fit(self, x_train, y_train):
        datasetLength = len(x_train)
        self.w = [1 / datasetLength for _ in range(datasetLength)]  #the initialization of the weights list
        self.h_t = []       #the initialization of the hypotheses list
        self.z = []     #the initialization of the hypotheses weights list
        self.keys = [i for i in range(1570)]
        
        i = 0
        while (i < self.m):
            ht = SingleDepthTree()
            ht.fit(self.w, x_train, y_train, self.keys)
            self.h_t.append(ht)
            algorithm_results = ht.predict(x_train)
            error = 0
            for j in range(len(y_train)):   #calculate the error using each examples weight
                if(y_train[j] != algorithm_results[j]):
                    error += self.w[j]
            if (error >= 0.5):  #if error is greater than 0.5, remove the feature from our list
                self.h_t.remove(ht)
                self.keys.remove(ht.feature)
                continue
            z = 0
            if (error != 0):    #set the weight of the voting of each hypothesis
                z = 0.5 * log2((1 - error)/ error)
            elif error == 0:
                z = 1
            elif error >= 0.5:
                z = 0
            self.z.append(z)
            for j in range(len(y_train)): #change weights based on the errors and their exp value
                if (algorithm_results[j] == y_train[j]):
                    self.w[j] = self.w[j] * exp(-z)
                else:
                    if (error != 0):
                        self.w[j] = self.w[j] * exp(z)
            self.normalizeWeights()
            i += 1

    def normalizeWeights(self): #function to normalize weights to sum up to 1
        sum_1 = sum(self.w)
        new_W = []
        for x in self.w:
            new_W.append(x / sum_1)
        self.w = new_W

    def predict(self, TestData):
        results = []
        for row in TestData:
            sum_0 = 0
            sum_1 = 0
            index = 0
            for h in self.h_t:
                prediction = h.predict_row(row)
                if (prediction == 0):       #prediction is the return classification of the given row from the tree
                    sum_0 += self.z[index]  #we add the hypothesis weight in these sums
                else:
                    sum_1 += self.z[index]
                index += 1
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