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
            currentLine.append(int(feature))
        features.append(currentLine)
        labels.append(line[len(line)-1])

def loadDict(fileName, totalWords=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        totalWords.append(line[0])
        

def getBestClassifier(emails, labels, w): 
    classifiers = []

    for wordIndex in range(0, 4003): 
        errorPositive = 0.0
        errorNegative = 0.0

        predictionOnPositive = []
        predictionOnNegative = []

        for email in emails:
            emailIndex = emails.index(email)
            #print("on email : " + str(emails.index(email)) + " word exists : " + str(email[wordIndex]) + ". Label is " + str(labels[emailIndex]))

            positiveWrong = False
            negativeWrong = False

            if int(email[wordIndex]) == int(1) and int(labels[emailIndex]) == int(-1):
                #increment the error Positive 
                #print('1a')
                errorPositive += w[emailIndex]
                positiveWrong = True
            elif int(email[wordIndex]) == int(0) and int(labels[emailIndex]) == int(1):
                #print('1b')
                errorPositive += w[emailIndex]
                positiveWrong = True


            if int(email[wordIndex]) == int(1) and int(labels[emailIndex]) == int(1):
                #increment the error negative
                #print('2a')
                errorNegative += w[emailIndex]
                negativeWrong = True

            elif int(email[wordIndex]) == int(0) and int(labels[emailIndex]) == int(-1):
                #print('2b')
                errorNegative += w[emailIndex]
                negativeWrong = True

        
            if positiveWrong: 
                predictionOnPositive.append(0)
            elif not positiveWrong:
                predictionOnPositive.append(1)


            if negativeWrong: 
                predictionOnNegative.append(0)
            elif not negativeWrong:
                predictionOnNegative.append(1)


        classifiers.append([errorPositive, wordIndex, 1, predictionOnPositive])
        classifiers.append([errorNegative, wordIndex, -1, predictionOnNegative])

    classifiers.sort(key = lambda x: x[0])

    
    return classifiers[0]
    
def getNewW(classifier, emails, labels, w):

    newZ = 0.0
    alpha = 0.5 * numpy.log((1.0-classifier[0])/classifier[0])

    for i in range(len(classifier[3])):
        if classifier[3][i] == 1 : 
            newZ += w[i] * math.exp(-1.0*alpha)
        else : 
            newZ += w[i] * math.exp(alpha)
            
    newW = []
    for i in range(len(classifier[3])):
        if classifier[3][i] == 1: 
            newW.append((w[i] * math.exp(-1.0*alpha))/newZ)
        else :  
            newW.append((w[i] * math.exp(alpha))/newZ)


    return newW

def initializeWeights(size): 
    w = []
    for weight in range(0, size) : 
        w.append(float(1)/size)

    return w


trainingFeatures = []
trainingLabels = []
words = []

#loading training data
loadData('pa5train.txt', trainingFeatures, trainingLabels)
loadDict('pa5dictionary.txt', words)

#initialize w to 1/n 
w = initializeWeights(len(trainingFeatures))

#first round of boosting
classifiers = []
for i in range(0,3):
    firstClassifier = getBestClassifier(trainingFeatures, trainingLabels, w)
    w = getNewW(firstClassifier, trainingFeatures, trainingLabels, w)
    classifiers.append(firstClassifier)
    print(firstClassifier)


# firstClassifier = getBestClassifier(trainingFeatures, trainingLabels, w)
# print(firstClassifier)


