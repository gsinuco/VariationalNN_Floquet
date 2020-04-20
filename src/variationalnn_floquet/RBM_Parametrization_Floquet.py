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
import matplotlib as mpl
#from scipy.stats import unitary_group

############### IMPORT WHAT WE NEED FROM THIS PROJECT #####################
import model as Model
import matrixmanipulation as mp

############### RUN THE MODEL ################################



# Find the full Floquet spectrum for a range of parameters
H     = Model.Hamiltonian()
N = 128
RabiCoupling = np.linspace(0.0,H.dim-1,H.dim)
epochs = range(N)
CentralFloquetGap  = np.zeros([N],dtype=np.float64)
CentralFloquetEVec = np.zeros([N,H.dim,2],dtype=np.complex128)
for i in [127]:#epochs:
    delta = 1.7# + 2.0*i/(1.0*N)     
    Omega = 10.0*i/N
    phase = np.arctan(1.0)/3.1341341
    H     = Model.Hamiltonian(delta,Omega,phase)
    E,U   = tf.linalg.eigh(H.H_TLS)    
    CentralFloquetGap[i] = - tf.math.real(E[int(H.dim/2 - 1)]).numpy() + tf.math.real(E[int(H.dim/2)]).numpy()
    CentralFloquetEVec[i,:,0] = U[:,int(H.dim/2 - 1)].numpy()#/U[int(H.dim/2 - 1),int(H.dim/2)].numpy()
    CentralFloquetEVec[i,:,1] = U[:,int(H.dim/2    )].numpy()#/U[int(H.dim/2 - 1),int(H.dim/2)].numpy()
#print(tf.math.real(E[int(model.dim/2 - 1)]).numpy()-tf.math.real(E[int(model.dim/2)]).numpy())

CentralFloquetNorm = np.sqrt(np.power(np.imag(CentralFloquetEVec),2) +np.power(np.real(CentralFloquetEVec),2))

CentralFloquetAtan = Model.phase(CentralFloquetEVec)

#CentralFloquetAtan = np.arctan(np.imag(CentralFloquetEVec)/np.real(CentralFloquetEVec)) 

index_ = 32

zerophase = np.arctan(np.imag(CentralFloquetEVec[index_,int(H.dim/2 - 1),:])/np.real(CentralFloquetEVec[index_,int(H.dim/2 - 1),:]))

CentralFloquetSin = np.imag(CentralFloquetEVec)/CentralFloquetNorm

CentralFloquetCos = np.real(CentralFloquetEVec)/CentralFloquetNorm

########## TRAINING AGAINST THE RWA ###########################
########## TO OBTAIN AN INITIAL SET RBM PARAMETERS ################
########## THIS IS EQUIVALENT TO RECONSTRUCT A SET OF WAVE FUNCTIONS ######
 

#mpl.pyplot.plot( RabiCoupling,   np.cos(CentralFloquetAtan[index_,:,0]),
#                -RabiCoupling,   np.cos(CentralFloquetAtan[index_,:,1]),
#                 RabiCoupling,   np.cos(CentralFloquetAtan[index_,:,0]-CentralFloquetAtan[index_,int(H.dim/2-1)-1,0]+phase))

model = Model.Model(delta,Omega,phase)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')
N = 512
epochs = range(N)
loss_list_RWA = tf.Variable([],dtype=tf.float64)
for i in epochs:
    loss_value, grads = Model.grad_fun(model,Model.loss_Floquet)
    loss_list_RWA = tf.concat([loss_list_RWA,[loss_value.numpy()]],axis=0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #if(i==0): print(tf.abs(model.UF))
    if(i == (N-1)):
        UF = Model.Unitary_Matrix(model) 
        print(tf.abs(model.UF))
        print(tf.abs(UF))        
        #print(tf.abs(model.U_RWA))
        #print(tf.math.imag(model.U_RWA)/tf.math.real(model.U_RWA)) 
        print("Final loss value (reconstruction): ",loss_value.numpy())

#mpl.pyplot.plot(loss_list_RWA)        

########## FULL TRAINING  ################
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
N = 512
epochs = range(N)
loss_list = tf.Variable([],dtype=tf.float64)
for i in epochs:
    loss_value, grads = Model.grad_fun(model,Model.loss)
    loss_list = tf.concat([loss_list,[loss_value.numpy()]],axis=0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if(i == (N-1)):
        UF = Model.Unitary_Matrix(model) 
##        print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))
        U_ = tf.abs(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
        #print(U_)
        print("Final loss value (Floquet): ",loss_value.numpy())    

#mpl.pyplot.plot(loss_list)



print("Test unitarity of the Floquet representation: ")    
UF = Model.Unitary_Matrix(model) 
print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))

print("Hamiltonian in the dressed basis: ")    
U_ = tf.math.real(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
print(U_)


print("Hamiltonian in the dressed basis: ")    
U_ = tf.math.real(tf.transpose(tf.math.conj(model.U_Floquet))@model.H_TLS@model.U_Floquet)
print(U_)

print("Floquet respresentation of dressed states: ")    
#print(tf.abs(UF))

mpl.pyplot.plot(range(model.dim),Model.Rect2Pol(model.U_Floquet)[0],Model.Rect2Pol(UF)[0]+1)
mpl.pyplot.plot(range(model.dim),Model.Rect2Pol(model.U_Floquet)[1],Model.Rect2Pol(UF)[1]+1)

#print(Model.loss(model))
#print(Model.grad_fun(model,Model.loss)[0])
#print(Model.loss_RWA(model))
#print(Model.grad_fun(model,Model.loss_RWA)[0])

















