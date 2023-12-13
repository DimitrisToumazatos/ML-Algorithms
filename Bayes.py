import csv
import os
import re
import collections

l = []
with open("most_common_words.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    l.append(row[0])

DIR = "aclImdb\\train\\neg\\"
good = []
list1 = os.listdir(DIR)
for filename in list1:
    words = re.findall(r'\w+', open("aclImdb\\train\\neg\\" + filename, encoding="utf8").read().lower())
    vec = []
    for i in l:
        if i in words:
          vec.append(1)
        else:
          vec.append(0)
    good.append(vec)

with open('negative.csv', 'w', newline='') as csv_file:  
    writer = csv.writer(csv_file)
    writer.writerows(good)


"""
class NaiveBayes:
   
   def __init__(self) -> None:
      pass
   """

