import collections
import math
import numpy

def loadData(fileName, features=[], labels=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        currentLine = []
        for feature in line[:-1]:
            currentLine.append(float(feature))
        features.append(currentLine)
        labels.append(line[len(line)-1])

def loadDict(fileName, totalWords=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        totalWords.append(line[0])
        

def getBestClassifier(words, emails, labels, w): 
    classifiers = []

    for word in words: 
        errorPositive = 0
        errorNegative = 0

        wordIndex = words.index(word)

        for email in emails:
            if email[wordIndex] == 1 and labels[emails.inedx(email)] == -1:
                #increment the error Positive 
            if email[wordIndex] == 1 and labels[emails.inedx(email)] == 1:
                #increment the error negative



trainingFeatures = []
trainingLabels = []
words = []

loadData('pa5train.txt', trainingFeatures, trainingLabels)
loadDict('pa5dictionary.txt', words)
print(words)