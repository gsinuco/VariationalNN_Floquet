#!/usr/bin/env python3
"""
Created on Wed Feb 12 10:44:59 2020

@author: German Sinuco

Skeleton modified from 
https://www.tensorflow.org/tutorials/customization/custom_training
https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
                       
Training of an RBM parametrization of the unitary matrix that diagonalises the extended HAMILTONIAN of harmonically driven quantum systems 
    
==================== IMPORTANT NOTE ========================
   
============================================================
"""


############### IMPORT WHAT WE NEED FROM THE OS/PYTHON DISTRIBUTION #####################
from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import tensorflow as tf
import numpy as np
import math as m
#from scipy.stats import unitary_group

############### IMPORT WHAT WE NEED FROM THIS PROJECT #####################
import model as Model
import matrixmanipulation as mp



############### SELECT AN OPTIMIZER  ################################
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)



 ############### RUN THE MODEL ################################
model = Model.Model()
#loss_value = Model.loss(model)
#print("Initial UF guess: ", mp.Unitary_Matrix(model))
#print("Initial loss value: ",loss_value.numpy())

#epochs = range(1)
#for i in epochs:
# #   loss_value, grads = Model.grad(model)
#    optimizer.apply_gradients(zip(grads, model.trainable_variables))
#    
#print("Final loss value: ",loss_value.numpy())    
#print("Final UF matrix:", mp.Unitary_Matrix(model))

UF       = mp.Unitary_Matrix(model)
U_       = tf.abs(tf.transpose(tf.math.conj(UF))@(model.H_TLS@UF))
U_diag   = tf.linalg.tensor_diag_part(U_)  
dotProd  = tf.math.reduce_sum(abs(U_),axis=1,)
residual = tf.math.reduce_sum(tf.pow((U_diag-dotProd),2),0)
print(residual)
print(tf.abs(UF))
print(U_)
 












