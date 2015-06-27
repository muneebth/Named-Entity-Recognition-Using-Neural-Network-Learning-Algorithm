import numpy as np
from data_utils import *
import random


def initialize_parameters_nn(input_size,hidden_size,output_size):


    model={}
    
    # W1 is  parameter weight  for the first layer of neural network including bias term as the parameter. shape (hidden_size,input_size+1)
    
    e_init     = np.sqrt(6) / np.sqrt(input_size + hidden_size)
    model["W1"]=np.random.randn(hidden_size,input_size+1)* 2*e_init - e_init
    """ W2 is  parameter weight  for the second layer of neural network including bias term as the parameter. shape           (output_size,hidden_size  +1)"""
    model["W2"]=np.random.randn(output_size,hidden_size+1)* 2*e_init - e_init
    
    return model
    
    
    
def calculate_gradient_loss(X,model,y=None,reg=.10):
    
    #unpacked the parameters from nn model dictionary
    W1,W2=model["W1"],model["W2"]
  
    X=np.array([np.concatenate((np.array([1]),X))]).T
   
   
    loss=0.0
    Z1=W1.dot(X)
    
    A1=np.zeros_like(Z1)
    A1=sigmoid(Z1)
    
    A1=np.concatenate((np.array([[1]]),A1),0)
    Z2=W2.dot(A1)
    
    # MAXIMUM VALUE IS SUBSTRACTED FROM ALL
    Z2-=np.max(Z2)
    
    Z2=np.exp(Z2)
    
    Z2/=np.sum(Z2)
    
    #print Z2.shape
    
    loss=-1*np.log(Z2[y])
    
    # loss=loss+0.5*reg*(np.sum(np.square(W1[:,1:]))+np.sum(np.square(W2[:,1:])))
    
    
    if(y==None):
       return loss
    
    grad={}
    
    dZ2=np.zeros_like(Z2)
    
    dZ2[:]=Z2
    dZ2[y]-=1

    dW2=dZ2.dot(A1.T)
    dA1=np.dot(W2.T,dZ2)
    #removing bias activation
    dA1=dA1[1:]   
    dZ1=dsigmoid(Z1)*dA1
    dW1=dZ1.dot(X.T)
    dx=np.dot(W1.T,dZ1) 
    dx=dx[1:]
    
    #ADDING REGULARIZATION TO WEIGHTS
    #dW1[:,1:]+=reg*W1[:,1:]
    # dW2[:,1:]+=reg*W2[:,1:]   
    
    
    grad["W1"]=dW1
    grad["W2"]=dW2
    grad["X"]=dx
    
    return loss,grad
    

def sigmoid(X):
    return 1/(1+np.exp(-X*1.000))
    

def dsigmoid(X):
     return sigmoid(X)*(1-sigmoid(X))    
    
      
    
