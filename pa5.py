import collections
import math
import numpy
from random import randint

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