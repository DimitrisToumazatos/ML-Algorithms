import re
import os
import csv

n = 100
m = 1900
k = 0


posNegList = ["pos", "neg"]
dict1 = {}
s1 = "aclImdb\\dev\\"
s2= "\\"

l =[]

with open('most_common_words.csv', 'r') as csv_file:        # read from the most common words file
    csvreader = csv.reader(csv_file)
    for i in csvreader:                   
        l.append(i[0])


for i in posNegList:            # create verctors for the traing data (positie/negative) 

    DIR = s1+i+s2
    dev = []
    list1 = os.listdir(DIR)     #a function which creates a list which contains the file names in the DIR folder 
    for filename in list1:
        words = re.findall(r'\w+', open(s1+i+s2 + filename, encoding="utf8").read().lower())
        vec = []
        for j in l:
            if j in words:
                vec.append(1)
            else:
                vec.append(0)
        dev.append(vec)

    if i == "pos":
        temp = "positiveDev.csv"
    else:
        temp = "negativeDev.csv"
        
    with open(temp, 'w', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerows(dev)


