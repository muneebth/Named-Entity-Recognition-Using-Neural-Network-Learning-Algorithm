
from nn_classifier import *
import numpy as np
from math import pow

from WindowModel import *
from WordMatrix import *
import time

dataDir = "data/"
dataFile = dataDir +"train"
vocab = dataDir +"vocab.txt"
testFile = dataDir+"dev"





vocabMatrix=WordMatrix()
contextSize=5

training_model=WindowModel( dataFile, vocabMatrix,(contextSize - 1)/2 )
training_tuples=training_model.generate_word_tuples()
Xtr_rows,Ytr=training_model.generate_word_vectors()
Ytr=Ytr.T
print 'training_info '
print Xtr_rows.shape
print Ytr.shape



test_model=WindowModel( testFile, vocabMatrix,(contextSize - 1)/2 )
test_tuples=test_model.generate_word_tuples()
Xte_rows,Yte=test_model.generate_word_vectors()
Yte=Yte.T
print 'test_info '
print Xte_rows.shape
print Yte.shape


Xte_rows=Xte_rows-np.mean(Xte_rows,0)
Xte_rows=Xte_rows/np.std(Xte_rows,0)

Xtr_rows=Xtr_rows-np.mean(Xtr_rows,0)
Xtr_rows=Xtr_rows/np.std(Xtr_rows,0)   


nn=NeuralClassifier()
for i in range(-3,-2,1):
  nn.train(Xtr_rows,Ytr,pow(10,i),.2) 
  Y_pred=np.zeros(Yte.shape[0],dtype=Ytr.dtype)
  Y_pred=nn.predict(Xte_rows)
  print "\n\n\n**************************************************************************************\n"
  print "prediction ",Y_pred[:100]
  print " Learning rate ",pow(10,i)
  print " Regularization  ",.1
  print " EFFICIENCY IN PREDICTION FOR ",np.mean(Y_pred==Yte)
  print "\n**************************************************************************************\n"
  prec_prob=float(np.sum(Y_pred[Y_pred==0]==Yte[Y_pred==0]))/np.sum(Y_pred==0)
  print prec_prob
  recall_prob=float(np.sum(Y_pred[Yte==0]==Yte[Yte==0]))/np.sum(Yte==0)
  print recall_prob
  F_score_prob=2*prec_prob*recall_prob/(prec_prob+recall_prob)


  print " Precision of other %f "%(prec_prob)
  print " Recall of other %f "%(recall_prob)
  print " F-score of other %f "%(F_score_prob)


  prec_test=float(np.sum(Y_pred[Y_pred==1]==Yte[Y_pred==1]))/np.sum(Y_pred==1)
  recall_test=float(np.sum(Y_pred[Yte==1]==Yte[Yte==1]))/np.sum(Yte==1)

  F_score_test=2*prec_test*recall_test/(prec_test+recall_test)


  print " Precision of Person %f "%(prec_test)
  print " Recall of Person %f "%(recall_test)
  print " F-score of Person %f "%(F_score_test)

  
  
  
