#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:45:37 2020

@author: German Sinuco
         gsinuco@gmail.com
         
Part of the project RBM parametrisation of Floquet systems.


Definition of the model: Definition of the Floquet Hamiltonian
                         RBM parametrisation of the micromotion operator
                         Definition of the loss function and its gradient
                         definition of the training sequence
                         
"""
import tensorflow as tf
import numpy as np
import math as m

import matrixmanipulation as mp



class Model(object):
  def __init__(self):

    # Hamiltonian parameters
    self.delta  = 1.00
    self.Omega  = 0.10

    # Initialize the spin value and number of floquet channels
    self.hidden_n  = 10  # hidden neurons
    self.hidden_ph = 10  # hidden neurons

    #self.S   = 4  # spin 3.2. Hilbert space dimension
    self.S   = 2     # spin 1/2. Hilbert space dimension
    self.N   = 1      # Number of positive Floquet manifolds
    self.dim = self.S*(2*self.N+1) # Dimension of the extended floquet space
    
    uf_      = tf.random.stateless_uniform([self.dim,self.dim],seed=[2,1],dtype=tf.float32,minval=0.0,maxval=1.0)      
    s,u,v    = tf.linalg.svd(uf_, full_matrices=True)
    #uf_      = u
    
    # Declaring training variables
    # Training parameters defining the norm 
    self.W_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n,self.dim*self.dim,3],
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True) 
    self.b_n   = tf.Variable(tf.random.stateless_uniform([self.dim*self.dim,3], 
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True)
    self.c_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n],                    
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True)
    
    # Training parameters defining the phase
    self.W_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph,self.dim*self.dim,3],
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True) 
    self.b_ph   = tf.Variable(tf.random.stateless_uniform([self.dim*self.dim,3], 
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True)
    self.c_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph],                    
                                                       seed=[1,1],dtype=tf.float32,
                                                       minval=0.0,maxval=1.0),trainable=True)
    
    
    UF_aux   = tf.Variable(np.zeros((self.dim*self.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator

    UF_n     = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
    UF_ph    = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
    self.UF  = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator

    
    # defining the labels of the input layer, which are the components of the UF matrix
    self.x = tf.Variable([[0.0,0.0,0.0]],dtype=tf.float32)
    counter = 0
    self.count = counter
    for l in range(-self.N,self.N+1):
        for i in range(1,self.S+1):        
            for j in range(1,self.S+1):
                if(self.S==4):
                    y = [[i-2.5,j-2.5,l]]
                if(self.S==2):
                    y = [[i-1.5,j-1.5,l]]
                self.x = tf.concat([self.x, y], 0) 
                counter +=1
                self.count = counter


    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    WX_n = [tf.reduce_sum(tf.multiply(self.x[1:counter+1],self.W_n[0]),1)+self.c_n[0]]
    for j in range(1,self.hidden_n):
        y = tf.reduce_sum(tf.multiply(self.x[1:counter+1],self.W_n[j]),1)+self.c_n[j]
        WX_n = tf.concat([WX_n, [y]], 0) 
        
    UF_aux = tf.sqrt(tf.abs(tf.multiply(tf.reduce_prod(tf.math.cosh(WX_n),0),tf.exp(
                                        tf.transpose(tf.reduce_sum(tf.multiply(
                                                self.x[1:counter+1],self.b_n),1))))))
    UF_n = tf.reshape(UF_aux,[self.dim,self.dim])
    
    # 2. Phase 
    
    WX_ph = [tf.reduce_sum(tf.multiply(self.x[1:counter+1],self.W_ph[0]),1)+self.c_ph[0]]
    for j in range(1,self.hidden_ph):
        y = tf.reduce_sum(tf.multiply(self.x[1:counter+1],self.W_ph[j]),1)+self.c_ph[j]
        WX_ph = tf.concat([WX_ph, [y]], 0) 
        
    UF_aux = tf.multiply(tf.reduce_prod(tf.math.cosh(WX_ph),0),tf.exp(
            tf.transpose(tf.reduce_sum(tf.multiply(self.x[1:counter+1],self.b_ph),1))))
    UF_ph = tf.reshape(tf.math.log(UF_aux),[self.dim,self.dim])
    
    UF_cos = tf.cos(UF_ph/2.0)
    UF_sin = tf.sin(UF_ph/2.0)
    
    UF = tf.complex(UF_n*UF_cos,UF_n*UF_sin)

    # 1st of March 2020. Task: REVISE NORMALISATION AND GRAM-SCHMIDT PROCEDURE FOR COMPLEX VECTORS
    # 5th of March 2020. Normalisation done by hand: OK. Now I am using the G-S algorithm 
    #                    reported in  https://stackoverflow.com/questions/48119473/gram-schmidt-orthogonalization-in-pure-tensorflow-performance-for-iterative-sol. 
    #                    Task: incorparate a basis rotation in the training loop
    
    UF = mp.normalisation(UF)
    UF = mp.tf_gram_schmidt(UF)

    self.UF = UF
    
    if self.S == 2:
        # spin 1/2 
        Identity   =     tf.constant([[1.0,0.0],[ 0.0, 1.0]],dtype = tf.complex64)
        Sx         = 0.5*tf.constant([[0.0,1.0],[ 1.0, 0.0]],dtype = tf.complex64)
        Sz         = 0.5*tf.constant([[1.0,0.0],[ 0.0,-1.0]],dtype = tf.complex64)
    else:
        if self.S == 4:
        # spin 3/2
            Identity   =     tf.constant([[1.0,0.0,0.0,0.0],
                                  [0.0,1.0,0.0,0.0],
                                  [0.0,0.0,1.0,0.0],
                                  [0.0,0.0,0.0,1.0]],dtype = tf.complex64)
            Sx         = 0.5*tf.constant([[0.0,         np.sqrt(3.0),0.0,          0.0],
                                  [np.sqrt(3.0),0.0,         np.sqrt(4.0), 0.0],
                                  [0.0,         np.sqrt(4.0),0.0,          np.sqrt(4.0)],
                                  [0.0,         0.0,         np.sqrt(3.0), 0.0]],dtype = tf.complex64)
            Sz         = 0.5*tf.constant([[3.0,0.0, 0.0, 0.0],
                                  [0.0,1.0, 0.0, 0.0],
                                  [0.0,0.0,-1.0, 0.0],
                                  [0.0,0.0, 0.0,-3.0]],dtype = tf.complex64)
    #else:
    #    if (self.S != 4 & self.S !=2):
    #        for j in range(0,self.S):
    #            H[j,j]
    
    #    for i in range(0,2*self.N): 
    #  i_r = self.S*(i) 
    #  j_r = i_r + self.S 
      
    #  i_c = i_r + self.S 
    #  j_c = i_c + self.S 
    #  self.H[i_r:j_r,i_c:j_c].assign(coupling)  
    #  self.H[i_c:j_c,i_r:j_r].assign(coupling)  
    
    #for i in range(0,2*self.N+1): 
    #  i_ = self.S*(i) 
    #  j_ = i_ + self.S 
    #  self.H[i_:j_,i_:j_].assign(H_qubit + (-self.N + i)*omega*Identity)  #self.UF[0,0]

    
    
    self.H_TLS = tf.Variable(self.delta*Sz+0.5*self.Omega*Sx,shape=(self.dim,self.dim),dtype = tf.complex64,trainable = False)  # ext. Hamiltonian
        
    self.trainable_variables = [self.W_n,self.b_n,self.c_n,self.W_ph,self.b_ph,self.c_ph]
    
    
    
  def getH(self):
    return self.H_TLS
    
  def __call__(trainable_variables):     
    return self.H_TLS




def train(model,learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model)
  dU = t.gradient(current_loss, model.trainable_variables)
  model.UF.assign_sub(learning_rate*dU)

# 3e. Loss function := Use U^dagger H U, sum over the columns, take the difference with the diagonal, 
#     the loss function is the summ of the square of these differences. 

def loss(model):
  # define the loss function explicitly including the training variables: self.W, self.b
  # model.UF is a function of self.W,self.b,self.c
    #UF    = tf.Variable(np.zeros((model.dim*model.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
  
    UF = tf.Variable(np.zeros((model.dim,model.dim),dtype=np.complex64)) 
    a  = np.zeros((model.dim,model.dim),dtype=np.float32)
    counter = model.count 

    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    UF = mp.Unitary_Matrix(model)
    
    U_ = tf.abs(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
    U_diag = tf.linalg.tensor_diag_part(U_)  
    dotProd = tf.math.reduce_sum(abs(U_),axis=1,)    
    residual = tf.math.reduce_sum(tf.pow((U_diag-dotProd),2),0)
        
    U_ = tf.abs(tf.transpose(tf.math.conj(UF))@UF)
    #print(U_)
    U_diag = tf.linalg.tensor_diag_part(U_)  

    dotProd = tf.math.reduce_sum(abs(U_),axis=1)    

    residual_unitary = tf.pow(tf.math.reduce_sum(dotProd,0) - model.dim,2.0)
    
    #residual += 1.0*residual_unitary

    return residual 

# This is the gradient of the loss function. required for keras optimisers
def grad(model):
  with tf.GradientTape() as tape:
    loss_value = loss(model)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

