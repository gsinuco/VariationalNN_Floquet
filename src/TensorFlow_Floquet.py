#!/usr/bin/env python3
"""

modified from https://www.tensorflow.org/tutorials/customization/custom_training
              https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
Created on Sat Aug 10 11:30:57 2019
@author: German Sinuco

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import math as m
from scipy.stats import unitary_group

class Model(object):
  def __init__(self):
    # Initialize the spin value and number of floquet channels
    self.S     = 2                   # spin
    self.N     = 0                   # floquet manifolds
    self.dim   = self.S*(2*self.N+1) # dimension of the extended floquet space
    aux_norm   = tf.random.stateless_uniform([self.dim,self.dim],seed=[1,1],dtype=tf.float32,minval=0.0,maxval=1.0)      
    aux_phase  = tf.random.stateless_uniform([self.dim,self.dim],seed=[3,9],dtype=tf.float32,minval=-m.pi*1.0,maxval=m.pi*1.0)          
    uf_        = tf.complex(aux_norm*tf.cos(aux_phase),aux_norm*tf.sin(aux_phase))
    s,u,v      = tf.linalg.svd(uf_, full_matrices=True)
    uf_        = u
    self.UF_A  = tf.Variable(aux_norm, dtype = tf.float32,trainable = True)  # ext. micromotion operator amplitude
    self.UF_ph = tf.Variable(aux_phase,dtype = tf.float32,trainable = True)  # ext. micromotion operator phase
    self.UF    = tf.Variable(uf_,dtype = tf.complex64,trainable = False)     # ext. micromotion operator
    self.H     = tf.Variable(0.0*uf_,shape=(self.dim,self.dim),dtype = tf.complex64,trainable = False)  # ext. Hamiltonian
    self.H_aux = tf.Variable(uf_,shape=(self.dim,self.dim),dtype = tf.complex64,trainable = False)  # ext. aux operator
        
    coupling   = tf.constant([[0,1],[-1, 0]],dtype = tf.complex64)
    H_qubit    = tf.constant([[1,0],[0,-1]],dtype = tf.complex64)
    Identity   = tf.constant([[1,0],[0, 1]],dtype = tf.complex64)
    omega      = 0.9
      
    for i in range(0,2*self.N): 
      i_r = self.S*(i) 
      j_r = i_r + self.S 
      
      i_c = i_r + self.S 
      j_c = i_c + self.S 
      self.H[i_r:j_r,i_c:j_c].assign(coupling)  
      self.H[i_c:j_c,i_r:j_r].assign(coupling)  
    
    for i in range(0,2*self.N+1): 
      i_ = self.S*(i) 
      j_ = i_ + self.S 
      self.H[i_:j_,i_:j_].assign(H_qubit + (-self.N + i)*omega*Identity)  #self.UF[0,0]
        
    self.trainable_variables = [self.UF_A,self.UF_ph]
    
  def getH(self):
    return self.H
    
  def __call__(self):
      uf_      = tf.complex(self.UF_A*tf.cos(self.UF_ph),self.UF_A*tf.sin(self.UF_ph))
      uf__     = tf.square(tf.abs(tf.transpose(tf.math.conj(uf_))@(self.H@uf_)))
      trace_   = tf.linalg.trace(uf__)/(self.dim*self.dim)      
      residual = tf.sqrt(tf.reduce_mean(uf__) - trace_)      
      return residual
  


def loss(predicted_y):
  return tf.reduce_mean(tf.square(predicted_y))


def train(model, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model())
  gradients = t.gradient(current_loss,  model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  
def loss_():
    loss_var = tf.Variable(1.0, dtype = tf.float32,trainable = True) 
    return loss_var
  
model = Model()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
let = loss_
#callable(let)

epochs = range(0,500)
for epoch in epochs:
    optimizer.minimize(model,model.trainable_variables)

uf_ = tf.complex(model.UF_A*tf.cos(model.UF_ph),model.UF_A*tf.sin(model.UF_ph))
H   = tf.transpose(tf.math.conj(uf_))@model.H@uf_
print(abs(H))

#a=model.residual_fun()
#optimizer.get_gradients(model.residual_fun(),model.trainable_variables)
  
#optimizer.minimize(model(),model.trainable_variables)

# Collect the history of W-values and b-values to plot later
Hs, bs = [], []
epochs = range(0,500)
for epoch in epochs:
    #Hs.append(model.UF.numpy())
    #current_loss = loss(model())
    train(model, learning_rate=0.5)
    #print(current_loss.numpy())

uf_ = tf.complex(model.UF_A*tf.cos(model.UF_ph),model.UF_A*tf.sin(model.UF_ph))
H = tf.transpose(tf.math.conj(uf_))@model.H@uf_
print(abs(H))

#tf.keras.estimator.model_to_estimator()