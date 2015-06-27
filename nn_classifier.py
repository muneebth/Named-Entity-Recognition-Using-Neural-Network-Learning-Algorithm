import numpy as np
from NeuralNet import *
import random

class NeuralClassifier:
   
    def __init__(self):
        pass
        
    def train(self, X, y, learning_rate=1e-4, reg=1e-1, num_iters=100000, verbose=False):     
            
        num_train,dim=X.shape
        num_class=np.max(y)+1
        self.Xtr=X
        self.Ytr=y
        self.model=initialize_parameters_nn(dim,1000,num_class)
        total_loss=0.0
        shuffled_index=range(num_train)
        print " Number of training samples ", num_train
        for i in range(0,5):
          random.shuffle(shuffled_index)
          count=0
          for j in shuffled_index:
           count=count+1 
           loss,grad=calculate_gradient_loss(self.Xtr[j],self.model,self.Ytr[j],reg)
           self.model["W1"]-=learning_rate*grad["W1"]
           self.model["W2"]-=learning_rate*grad["W2"]
           #self.Xtr[j]-=learning_rate*grad["X"]
           #print loss
           total_loss+=loss
           if((count)%(num_train/4)==0):
              print " Iteration ",i
              print count," th loss of ",total_loss/1000
              print " Regularization ",reg
              print " learning rate ",learning_rate
              total_loss=0
              
            
    def predict(self,X):
        num_test,dim=X.shape
        print 'num_test ',num_test
        W1,W2=self.model["W1"],self.model["W2"]
        X=np.concatenate((np.ones((num_test,1)),X),1).T
        Z1=W1.dot(X)
        A1=np.zeros_like(Z1)
        A1=sigmoid(Z1)
        A1=np.concatenate((np.ones((1,num_test)),A1),0)
        Z2=W2.dot(A1)        
        return Z2.argmax(0)




