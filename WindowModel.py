from WordMatrix import *
import numpy as np
import scipy as sp
import pdb

dataDir = "data/"
dataFile = dataDir +"train"
wordVecs = dataDir +"wordVectors.txt"
vocab = dataDir +"vocab.txt"
wordDim = 50

class WindowModel():
    
    def __init__(self,fn, wm, windowPadding=1):
        self.filename = fn
        self.wordMatrix = wm
        self.windowPadding = windowPadding
        self.contextSize = windowPadding*2 +1
        self.dimSize = 50
        self.allSentences = []
        self.labels = []
        self.allTuples = []
        self._split_into_sentences()


    def _split_into_sentences(self):
        """
        Extract the labels and turn the words into sentences

        :return:
        """
        EOL_CHARS = ".!?"
        allSentences = []
        with open(self.filename, 'r') as f:
            sentence = []
            for line in f:
                try:
                    word, label = line.split()
                except:
                    continue
                #no the end of the sentence
                sentence += [word]
                intLabel = 1 if label == "PERSON" else 0
                self.labels += [intLabel]
                if word in EOL_CHARS:
                    allSentences += [sentence]
                    sentence = []

            #in case the last sentence doesn't end with proper punctuation!
            if sentence != []:
                allSentences += [sentence]
            self.allSentences = allSentences

    def generate_word_tuples(self):
        """

        :return:
        sentenceTups: A two dimensional list.  The first dimension indexes the sentences and the second gives the tuple
        self.labels =
        """
        indexDict = self.wordMatrix.indexDict
        start = indexDict['<s>']
        stop = indexDict['</s>']
        sentenceTups = []
        for sentence in self.allSentences:
            l = len(sentence)
            for ind in range(l):
                wordAndContext = []
                for j in range(ind-self.windowPadding, ind+self.windowPadding+1):
                    if j >= 0 and j < l:
                        currentWord = sentence[j][0]
                        wordIndex = indexDict[currentWord] if currentWord in indexDict else 0
                        wordAndContext += [wordIndex]
                    elif j>0:
                        wordAndContext += [start]
                    else:
                        wordAndContext += [stop]
                sentenceTups += [wordAndContext]
        
        ## SET SOME INSTANCE VARIABLES
        self.allTuples = sentenceTups
        if len(sentenceTups) != len(self.labels):
            print "PROBLEM: The number of tuples and labels do not match"
        return sentenceTups



    def generate_word_vectors(self):
        if self.allTuples == [] or self.labels == []:
            print "Generating Tuples Of Word Indices e.g: (W_(n-2), W_(n-1), W, W_(n+1), W_(n+2) where W is the training/test example..."
            self.generate_word_tuples()
        m = len(self.allTuples)
        windowSize = self.windowPadding*2+1
        n = self.dimSize *windowSize
        vocabMatrix = self.wordMatrix.wordMatrix
        trainingMat = np.zeros((m,n))
        for row,trEx in enumerate(self.allTuples):
            for start, wordInd in enumerate(trEx):
                wordVec = vocabMatrix[:,wordInd]
                trainingMat[row, start*self.dimSize: (start+1)*self.dimSize] = wordVec

        return trainingMat, np.array(self.labels)


            


if __name__ == "__main__":
    wordMat = WordMatrix()
    wm = WindowModel(dataFile, wordMat,1)
    trainingMatrix, labs = wm.generate_word_vectors()




            
