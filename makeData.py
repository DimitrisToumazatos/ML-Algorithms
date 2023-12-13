#import re
#import collections
#import os
#from operator import itemgetter
#import csv
"""
DIR = "aclImdb\\train\\all\\"
dict1 = {}
list1 = os.listdir(DIR)
for filename in list1:
    words = re.findall(r'\w+', open("aclImdb\\train\\all\\" + filename, encoding="utf8").read().lower())
    for word in words:
        if len(word)>2:
            if word in dict1:
                dict1[word] += 1
            else:
                dict1[word] = 1



#most_common = dict(sorted(dict1.items(), key=itemgetter(1), reverse=True)[-1:-200:-1])
most_common = dict(sorted(dict1.items(), key=itemgetter(1), reverse=True)[:1500])

"""

import csv

l = []
with open("most_common.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    if len(row)>0:
        l.append([row[0]])

l = l[29:]

with open('most_common_words.csv', 'w', newline='') as csv_file:  
    writer = csv.writer(csv_file)

    writer.writerows(l)

  

