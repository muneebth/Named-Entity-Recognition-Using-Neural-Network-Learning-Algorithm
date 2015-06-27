import scipy as sp
import numpy as np

dataDir = "data/"
dataFile = dataDir +"train"
wordVecs = dataDir +"wordVectors.txt"
vocab = dataDir +"vocab.txt"
wordDim = 50

class WordMatrix():
    """
    A class that is responsible for creating a matrix representing each word's (in the vocabulary) vectorized representation.  

    indexDict: string -> int, english word -> column index of wordMatrix
    wordMatrix: 2d numpy array that is wordDim x numWords.  Every column represents the vectorized implementation of a word
    """
    def __init__(self):
        self.indexDict = {}
        self.wordMatrix = self.generate_word_matrix()


    def generate_word_matrix(self):
        with open(wordVecs, 'r') as f, open(vocab, 'r') as g:
            for i,word in enumerate(g):
                self.indexDict[word.strip("\n")] = i
            numWords = i+1

            newMatrix = np.zeros( (wordDim,numWords))
            for i, vec in enumerate(f):
                features = np.array( vec.split())
                newMatrix[:,i] = features.T
            return newMatrix






