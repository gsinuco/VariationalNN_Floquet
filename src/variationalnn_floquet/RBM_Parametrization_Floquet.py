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

############### IMPORT WHAT WE NEED FROM THIS PROJECT #####################
import model as Model
import matrixmanipulation as mp
import NN_model as NN_Model

############### FULL RUN ################################
# 1. CALCULATE THE EXACT FLOQUET STATES
# 2. CALCULATE THE RBM REPRESENTATION



H = Model.Hamiltonian()

N = 32

CentralFloquetE        = np.zeros([N,2],dtype=np.float64)
CentralFloquetEVec     = np.zeros([N,H.dim,2],dtype=np.complex128)

CentralFloquetE_RBM    = np.zeros([N,2],dtype=np.float64)
CentralFloquetEVec_RBM = np.zeros([N,H.dim,2],dtype=np.complex128)

loss_list_RWA          = tf.Variable([],dtype=tf.float64)

N_training = 32
N_tr_ph    = 2
N_tr_rho   = 2

file = open('RBM_TrainingFloquet.dat','wb')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for i in [16]:#range(N):
    
    delta = 1.7
    Omega = 10.0*i/N
    phase = 0.0#np.arctan(1.0)/4.1341341
    

########## TRAINING AGAINST THE RWA ###########################
########## TO OBTAIN AN INITIAL SET RBM PARAMETERS ################
########## THIS IS EQUIVALENT TO RECONSTRUCT A SET OF WAVE FUNCTIONS ######
 

    model = Model.RBM_Model(delta,Omega,phase)
    
    CentralFloquetEVec[i,:,0] = model.U_Floquet[:,0].numpy()
    CentralFloquetEVec[i,:,1] = model.U_Floquet[:,1].numpy()
    CentralFloquetE[i,0]      = tf.math.real(model.E_Floquet[0]).numpy()
    CentralFloquetE[i,1]      = tf.math.real(model.E_Floquet[1]).numpy() 

    for j in range(N_training):#[0,1,2,3,5]:
        # Fit the norm        
        for i_ in range(N_tr_rho):
            loss_value, grads = Model.grad_fun(model,Model.loss_RWA)

            loss_list_RWA = tf.concat([loss_list_RWA,[loss_value.numpy()]],axis=0)
            #optimizer.apply_gradients(zip(grads, model.trainable_variables))
            optimizer.apply_gradients(zip(grads[0:3], model.trainable_variables[0:3]))
        # Fit the phase
        for i_ in range(N_tr_ph):
            loss_value, grads = Model.grad_Phase(model,Model.loss_Floquet_Phase)
            loss_list_RWA = tf.concat([loss_list_RWA,[loss_value.numpy()]],axis=0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables[3:6]))


    # Modelling the phase distribution (0,pi) as a neural network
    # x = model.x[1:133,:].numpy()
    # y = tf.math.sign(tf.math.real(tf.reshape(model.U_Floquet,[2*model.dim]))).numpy() 
    # NNmodel, score  = NN_Model.NN_model(x,y)

    UF = Model.Unitary_Matrix(model) 
    print("Hamiltonian in the dressed basis (RBM parametrisation): ")    
    U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
    print(U_)

    if (UF.shape[1] > model.S):
        index_ = int(model.S*((UF.shape[1]/model.S -1))/2)
    else:   
        index_ = 0
        
    CentralFloquetEVec_RBM[i,:,0] = UF[:,index_].numpy()
    CentralFloquetEVec_RBM[i,:,1] = UF[:,index_+1].numpy()
    CentralFloquetE_RBM[i,0] = U_[0,index_].numpy()
    CentralFloquetE_RBM[i,1] = U_[1,index_+1].numpy() 


    print("Hamiltonian in the dressed basis (Floquet): ")    
    U_ = tf.math.real(tf.transpose(tf.math.conj(model.U_Floquet))@model.H_TLS@model.U_Floquet)
    print(U_)
    
    
    #np.savetxt(file, [model.delta,model.Omega,model.phase])
    #np.savetxt(file, [model.hidden_n,model.hidden_ph])
    #np.savetxt(file, [N_training ,N_tr_ph, N_tr_rho])
    #np.savetxt(file, [model.W_n.numpy()])
    #np.savetxt(file, [model.b_n.numpy()])
    #np.savetxt(file, [model.c_n.numpy()])
    #np.savetxt(file, [model.W_ph.numpy()])
    #np.savetxt(file, [model.b_ph.numpy()])
    #np.savetxt(file, [model.c_ph.numpy()])
    #np.savetxt(file, [model.U_Floquet.numpy()])
    #np.savetxt(file, [model.E_Floquet.numpy()])
    #np.savetxt(file, [CentralFloquetEVec_RBM[i,:,:]])
    #np.savetxt(file, [CentralFloquetE_RBM])
    



####################################################################
####################################################################
####################################################################
####################################################################

####################################################################
####### FINDING THE FLOQUET OPERATOR TRAINING A RBM   ##############
####################################################################

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_list = tf.Variable([],dtype=tf.float64)

CentralFloquetE_RBM2    = np.zeros([N,2],dtype=np.float64)
CentralFloquetEVec_RBM2 = np.zeros([N,H.dim,2],dtype=np.complex128)
trained_parameters_0 = [model.W_n,model.b_n,model.c_n,model.W_ph,model.b_ph,model.c_ph]




#%%
f = 1.0
loss_list = tf.Variable([],dtype=tf.float64)
for i in range(1):

    delta = 1.7
    Omega = 10.0*(i+16)/N 
    phase = 0.0#np.arctan(1.0)/4.1341341

    N_training = 128
    #trained_parameters = model.trainable_variables
    model = Model.RBM_Model(delta,Omega,phase,trained_parameters_0)
    #model = Model.RBM_Model(delta,Omega,phase)
    #init_parameters = [f*model.W_n,f*model.b_n,f*model.c_n,f*model.W_ph,f*model.b_ph,f*model.c_ph]
    
    #test_parameters_ = init_parameters + trained_parameters_0 
    
    #model = Model.RBM_Model(delta,Omega,phase,test_parameters_)
    UF_0  = Model.Unitary_Matrix(model) 
    
    CentralFloquetE[i+16,0]      = tf.math.real(model.E_Floquet[0]).numpy()
    CentralFloquetE[i+16,1]      = tf.math.real(model.E_Floquet[1]).numpy() 
    CentralFloquetEVec[i+16,:,0] = model.U_Floquet[:,0].numpy()
    CentralFloquetEVec[i+16,:,1] = model.U_Floquet[:,1].numpy()

    for i_ in range(N_training):
        loss_value, grads = Model.grad_fun(model,Model.loss)
        loss_list = tf.concat([loss_list,[loss_value.numpy()]],axis=0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if(i_ == (N_training-1)):
            UF = Model.Unitary_Matrix(model) 
            ##        print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))
            U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
            print(U_)
            print("Final loss value (Floquet): ",loss_value.numpy())    
    
    CentralFloquetE_RBM2[i,0]      = tf.math.real(U_[0,0]).numpy()
    CentralFloquetE_RBM2[i,1]      = tf.math.real(U_[1,1]).numpy() 
    CentralFloquetEVec_RBM2[i,:,0] = UF[:,0].numpy()
    CentralFloquetEVec_RBM2[i,:,1] = UF[:,1].numpy()

#mpl.pyplot.plot(loss_list)
    
    
    
    
    
#%%


#print("Test unitarity of the Floquet representation: ")    
#UF = Model.Unitary_Matrix(model) 
#print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))

#print("Hamiltonian in the dressed basis: ")    
#U_ = tf.math.real(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
#print(U_)

#print("Hamiltonian in the dressed basis: ")    
#U_ = tf.math.real(tf.transpose(tf.math.conj(model.U_Floquet))@model.H_TLS@model.U_Floquet)
#print(U_)

#print("Floquet respresentation of dressed states: ")    
#print(tf.abs(UF))

#mpl.pyplot.plot(range(model.dim),Model.Rect2Pol(model.U_Floquet)[0],Model.Rect2Pol(UF)[0]+1)

#mpl.pyplot.plot(range(model.dim),Model.Rect2Pol(model.U_Floquet)[1],Model.Rect2Pol(UF)[1]+1)

#print(Model.loss(model))
#print(Model.grad_fun(model,Model.loss)[0])
#print(Model.loss_RWA(model))
#print(Model.grad_fun(model,Model.loss_RWA)[0])

















