#!/usr/bin/env python3
"""
Created on Wed Feb 12 10:44:59 2020

@author: German Sinuco

Skeleton modified from 
https://www.tensorflow.org/tutorials/customization/custom_training
https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
                       
Training of an RBM parametrization of the unitary matrix that diagonalises the  2x2 real,
and symmetric HAMILTONIAN:
    
==================== IMPORTANT NOTE ========================
    as V2, but using complex parameters, which I used for the first time in TensorFlow_Floquet.py
============================================================
"""


from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import tensorflow as tf
import numpy as np
import math as m
from model import FloquetHamiltonian
from scipy.stats import unitary_group

class Model(object):
  def __init__(self,delta=0.0,Omega=0.1,phase=0.0):

    self.spin    = True    
    self.omega_0 = 1.00


      
      # Hamiltonian parameters
    self.delta  = 0.00
    self.omega  = self.delta + self.omega_0
    self.Omega  = 1.00
    self.phase  = phase # the phase in cos(omega t + phase)

    # Initialize the spin value and number of floquet channels
    self.hidden_n  = 4  # hidden neurons
    self.hidden_ph = 4  # hidden neurons

    self.S   = 4  # spin 3.2. Hilbert space dimension
    #self.S   = 2     # spin 1/2. Hilbert space dimension
    self.N   = 0      # Number of positive Floquet manifolds
    self.dim = self.S*(2*self.N+1) # Dimension of the extended floquet space
 
    zero_ = tf.constant(0.0,dtype=tf.float64)
    one_  = tf.constant(1.0,dtype=tf.float64)
    j_ = tf.constant(tf.complex(zero_,one_),dtype=tf.complex128)
    
    #uf_      = tf.random.stateless_uniform([self.dim,self.dim],seed=[2,1],dtype=tf.float32,minval=0.0,maxval=1.0)      
    #s,u,v    = tf.linalg.svd(uf_, full_matrices=True)
    #uf_      = u
    
    # Declaring training variables
    # Training parameters defining the norm 
    self.W_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n,self.dim*self.dim,2],
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True) 
    self.b_n   = tf.Variable(tf.random.stateless_uniform([self.dim*self.dim,2], 
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True)
    self.c_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n],                    
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True)
    
    # Training parameters defining the phase
    self.W_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph,self.dim*self.dim,2],
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True) 
    self.b_ph   = tf.Variable(tf.random.stateless_uniform([self.dim*self.dim,2], 
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True)
    self.c_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph],                    
                                                       seed=[1,1],dtype=tf.float64,
                                                       minval=-1.0,maxval=1.0),trainable=True)
    
    
    UF_aux  = tf.Variable(np.zeros((self.dim*self.dim), dtype=np.complex128),trainable = False)  # ext. micromotion operator

    UF_n  = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex128),trainable = False)  # ext. micromotion operator
    UF_ph  = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex128),trainable = False)  # ext. micromotion operator
    self.UF  = tf.Variable(np.zeros((self.dim,self.dim), dtype=np.complex128),trainable = False)  # ext. micromotion operator

    
    # defining the labels of the input layer, which are the components of the UF matrix
    self.x = tf.Variable([[0.0,0.0]],dtype=tf.float64)
    counter = 0
    self.count = counter
    for i in range(1,self.dim+1):        
        for j in range(1,self.dim+1):
            if(self.S==4):
                y = [[i-2.5,j-2.5]]
            if(self.S==2):
                y = [[i-1.5,j-1.5]]
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
    
    UF =  normalisation(UF)
    UF = tf_gram_schmidt(UF)

    self.UF = UF
    
    if self.S == 2:
        # spin 1/2 
        self.Identity   =     tf.constant([[1.0,0.0],[ 0.0, 1.0]],dtype = tf.complex128)
        self.Sx         =    0.5*tf.constant([[0.0,1.0],[ 1.0, 0.0]],dtype = tf.complex128)
        self.Sy         = j_*0.5*tf.constant([[0.0,1.0],[-1.0, 0.0]],dtype = tf.complex128)
        self.Sz         =    0.5*tf.constant([[1.0,0.0],[ 0.0,-1.0]],dtype = tf.complex128)
    else:
        if self.S == 4:
        # spin 3/2
            self.Identity   =     tf.constant([[1.0,0.0,0.0,0.0],
                                  [0.0,1.0,0.0,0.0],
                                  [0.0,0.0,1.0,0.0],
                                  [0.0,0.0,0.0,1.0]],dtype = tf.complex128)
            self.Sx         = 0.5*tf.constant([[0.0,         np.sqrt(3.0),0.0,          0.0],
                                  [np.sqrt(3.0),0.0,         np.sqrt(4.0), 0.0],
                                  [0.0,         np.sqrt(4.0),0.0,          np.sqrt(4.0)],
                                  [0.0,         0.0,         np.sqrt(3.0), 0.0]],dtype = tf.complex128)
            self.Sz         = 0.5*tf.constant([[3.0,0.0, 0.0, 0.0],
                                  [0.0,1.0, 0.0, 0.0],
                                  [0.0,0.0,-1.0, 0.0],
                                  [0.0,0.0, 0.0,-3.0]],dtype = tf.complex128)
    
    self.Szero      = tf.zeros([self.S,self.S],dtype=tf.complex128)

    #else:
    #    if (self.S != 4 & self.S !=2):
    #        for j in range(0,self.S):
    #            H[j,j]
    
    if self.N == 0:
        self.H_TLS = tf.Variable(self.delta*self.Sz+0.5*self.Omega*self.Sx,shape=(self.dim,self.dim),dtype = tf.complex128,trainable = False)  # ext. Hamiltonian
    else:
        self.H_TLS = FloquetHamiltonian(self)  # ext. Hamiltonian
        
    self.trainable_variables = [self.W_n,self.b_n,self.c_n,self.W_ph,self.b_ph,self.c_ph]
    
    
    
  def getH(self):
    return self.H_TLS
    
  def __call__(trainable_variables):     
    return self.H_TLS



def normalisation(U_):
    # U_  (in) original matrix
    #     (out)  matrix with normalised vectors
    normaU_ = tf.sqrt(tf.math.reduce_sum(tf.multiply(tf.math.conj(U_),U_,1),axis=0)) 
    U_ = tf.math.truediv(U_,normaU_)
    return U_

def tf_gram_schmidt(vectors):
    # add batch dimension for matmul
    basis = tf.expand_dims(vectors[:,0]/tf.norm(vectors[:,0]),0)
    for i in range(1,vectors.shape[0]):#vectors.get_shape()[0].value):
        v = vectors[:,i]
        # add batch dimension for matmul
        v = tf.expand_dims(v,0) 
        w = v - tf.matmul(tf.matmul(v, basis, adjoint_b=True), basis)
         # I assume that my matrix is close to orthogonal
        basis = tf.concat([basis, w/tf.norm(w)],axis=0)
    return basis




def Unitary_Matrix(model): 

    UF    = tf.Variable(np.zeros((model.dim*model.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator

    UF_n   = tf.Variable(np.zeros((model.dim,model.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
    UF_ph  = tf.Variable(np.zeros((model.dim,model.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
    
    #dim    = model.dim
    counter = model.count
    

    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    WX_n = [tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_n[0]),1)+model.c_n[0]]
    for j in range(1,model.hidden_n):
        y = tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_n[j]),1)+model.c_n[j]
        WX_n = tf.concat([WX_n, [y]], 0) 
        
    UF_n = tf.sqrt(tf.multiply(tf.reduce_prod(tf.math.cosh(WX_n),0),tf.transpose(tf.exp(tf.reduce_sum(
            tf.multiply(model.x[1:counter+1],model.b_n),1)))))
    UF_n = tf.reshape(UF_n,[model.dim,model.dim])
    
    # 2. Phase 
    WX_ph = [tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_ph[0]),1)+model.c_ph[0]]
    for j in range(1,model.hidden_ph):
        y = tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_ph[j]),1)+model.c_ph[j]
        WX_ph = tf.concat([WX_ph, [y]], 0) 
        
    UF_ph = tf.multiply(tf.reduce_prod(tf.math.cosh(WX_ph),0),tf.transpose(tf.exp(tf.reduce_sum(
            tf.multiply(model.x[1:counter+1],model.b_ph),1))))
    UF_ph = tf.reshape(tf.math.log(UF_ph),[model.dim,model.dim])
    
    
    UF_cos = tf.cos(UF_ph/2.0)
    UF_sin = tf.sin(UF_ph/2.0)
    
    UF = tf.complex(UF_n*UF_cos,UF_n*UF_sin)
    UF = tf_gram_schmidt(UF)

    #s,u,v    = tf.linalg.svd(UF, full_matrices=True)
    #UF = u
    
    return UF

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
    UF = Unitary_Matrix(model)
    
    U_ = tf.abs(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
    U_diag = tf.linalg.tensor_diag_part(U_)  
    dotProd = tf.math.reduce_sum(abs(U_),axis=1,)    
    residual = tf.math.reduce_sum(tf.abs((U_diag-dotProd)),0)
        
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


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)



 
model = Model()
loss_value = loss(model)
print("Initial UF guess: ", Unitary_Matrix(model))
print("Initial loss value: ",loss_value.numpy())

epochs = range(2048)
for i in epochs:
    loss_value, grads = grad(model)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
print("Final loss value: ",loss_value.numpy())    
print("Final UF matrix:", Unitary_Matrix(model))

UF       = Unitary_Matrix(model)
U_       = tf.abs(tf.transpose(tf.math.conj(UF))@(model.H_TLS@UF))
U_diag   = tf.linalg.tensor_diag_part(U_)  
dotProd  = tf.math.reduce_sum(abs(U_),axis=1,)
residual = tf.math.reduce_sum(tf.pow((U_diag-dotProd),2),0)
print(residual)
print(tf.abs(UF))
print(U_)
 












