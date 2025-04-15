#
#     TEXT ANLYSER
#

from collections import Counter
import matplotlib.pyplot as plt


def anaylseText(text):
     words = text.split()
     wordFrequency = Counter(words)
     uniqueWords = set(words)

     return [wordFrequency, uniqueWords]

def outputAnalysis(wordFrequency, uniqueWords):
     return f""" 
     Word Frequency: {wordFrequency}
     Unique Words: {uniqueWords}
     """

def plotAnalysis(wordFrequency, uniqueWords):
     freq = wordFrequency.values()
     cate = wordFrequency.keys()

     plt.bar(cate, freq)
     plt.xlabel('Words')
     plt.ylabel('Frequency')
     plt.title('Frequency of Words')
     plt.show()

userInput = input("Input text to be analysed: ")
output = anaylseText(userInput)

print(outputAnalysis(output[0], output[1]))
plotAnalysis(output[0], list(output[1]))