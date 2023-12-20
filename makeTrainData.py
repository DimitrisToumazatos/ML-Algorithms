import re
import os
from operator import itemgetter
import csv

n = 130 #For bayes algorithm we want it 130 for 80% accuracy
m = 1700  #For bayes algorithm we want it 1700 for 80% accuracy
k = 0


posNegList = ["neg", "pos"]
dict1 = {}
s1 = "aclImdb\\train\\"
s2= "\\"
for i in posNegList:                    # read training data and calculate count per word 
    DIR = s1+i+s2
    list1 = os.listdir(DIR)
    for filename in list1:
        words = re.findall(r'\w+', open(s1+i+s2 + filename, encoding="utf8").read().lower())
        for word in words:
            if len(word)>2:
                if word in dict1:
                    dict1[word] += 1
                else:
                    dict1[word] = 1

most_common = dict(sorted(dict1.items(), key=itemgetter(1), reverse=True)[:m]) # create and sort dictionary of most common words and their count 


l = list(most_common.keys())[n:]



with open('most_common_words.csv', 'w', newline='', encoding="utf8") as csv_file:        # write the most common words in a file
    writer = csv.writer(csv_file)
    for i in l:                   
        writer.writerow([i])


for i in posNegList:            # create verctors for the traing data (positie/negative) 

    DIR = s1+i+s2
    train = []
    list1 = os.listdir(DIR)     #a function which creates a list which contains the file names in the DIR folder 
    for filename in list1:
        words = re.findall(r'\w+', open(s1+i+s2 + filename, encoding="utf8").read().lower())
        vec = []
        for j in l:
            if j in words:
                vec.append(1)
            else:
                vec.append(0)
        train.append(vec)

    if i == "pos":
        temp = "positive.csv"
    else:
        temp = "negative.csv"
        
    with open(temp, 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerows(train)


