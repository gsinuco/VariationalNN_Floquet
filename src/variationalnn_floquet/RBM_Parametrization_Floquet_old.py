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
#import matplotlib as mpl
import matplotlib.pyplot as plt

############### IMPORT WHAT WE NEED FROM THIS PROJECT #####################
import model as Model
import matrixmanipulation as mp
import NN_model as NN_Model
#import Training as training

############### FULL RUN ################################
# 1. CALCULATE THE EXACT FLOQUET STATES
# 2. CALCULATE THE RBM REPRESENTATION

H = Model.Hamiltonian()

N = 32

CentralFloquetE        = np.zeros([N,2],dtype=np.float64)
CentralFloquetEVec     = np.zeros([N,H.dim,2],dtype=np.complex128)

CentralFloquetE_RBM    = np.zeros([N,2],dtype=np.float64)
CentralFloquetEVec_RBM = np.zeros([N,H.dim,2],dtype=np.complex128)


N_training_RWA = 4
N_training_Matrix = 512
N_tr_ph    = 2
N_tr_rho   = 2

#file = open('RBM_TrainingFloquet.dat','a')
file = 'RBM_TrainingFloquet_WaveFunction.dat'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


script_params ='#RWA Training: keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999,                                      epsilon=1e-07, amsgrad=False,name=Adam)'+ str([N_training_RWA, N_tr_ph, N_tr_rho])+ 'Matrix Diagonalisation: optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999,                                          epsilon=1e-07, amsgrad=False,name=Adam)'+str(N_training_Matrix)


Model.save_RBM(file,Script_Params=script_params)

for i in range(32):#[16]:#:,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32]:#range(N):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, 
                                     epsilon=1e-07, amsgrad=False,name='Adam')
    
    N_training = N_training_RWA
    
    delta = 1.7
    Omega = 10.0*i/N
    phase = np.arctan(1.0)/4.1341341
    
    loss_list_RWA          = tf.Variable([],dtype=tf.float64)
    
########## TRAINING AGAINST THE RWA ###########################
########## TO OBTAIN AN INITIAL SET RBM PARAMETERS ################
########## THIS IS EQUIVALENT TO RECONSTRUCT A SET OF WAVE FUNCTIONS ######
 

    model = Model.RBM_Model(delta,Omega,phase)
    
    CentralFloquetEVec[i,:,0] = model.U_Floquet[:,0].numpy()
    CentralFloquetEVec[i,:,1] = model.U_Floquet[:,1].numpy()
    CentralFloquetE[i,0]      = tf.math.real(model.E_Floquet[0]).numpy()
    CentralFloquetE[i,1]      = tf.math.real(model.E_Floquet[1]).numpy() 

    Model.save_RBM(file,Model=model)
    for j in range(N_training):#[0,1,2,3,5]:

        # Fit the norm        
        for i_ in range(N_tr_rho):
            #loss_value, grads = Model.grad_fun(model,Model.loss_Psi_rho)
            loss_value, grads = Model.grad_fun(model,Model.loss_RWA)
            #loss_value, grads = Model.grad_fun(model,Model.loss_Floquet)
            loss_list_RWA = tf.concat([loss_list_RWA,[loss_value.numpy()]],axis=0)
            #optimizer.apply_gradients(zip(grads, model.trainable_variables))
            optimizer.apply_gradients(zip(grads[0:3], model.trainable_variables[0:3]))

        # Fit the phase
        for i_ in range(N_tr_ph):
            #loss_value, grads = Model.grad_Phase(model,Model.loss_Psi_Phase)
            loss_value, grads = Model.grad_Phase(model,Model.loss_RWA_Phase)
            #loss_value, grads = Model.grad_Phase(model,Model.loss_Floquet_Phase)
            loss_list_RWA = tf.concat([loss_list_RWA,[loss_value.numpy()]],axis=0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables[3:6]))


    # Modelling the phase distribution (0,pi) as a neural network
    # x = model.x[1:133,:].numpy()
    # y = tf.math.sign(tf.math.real(tf.reshape(model.U_Floquet,[2*model.dim]))).numpy() 
    # NNmodel, score  = NN_Model.NN_model(x,y)

    UF = Model.Unitary_Matrix(model) 
    #print("Hamiltonian in the dressed basis (RBM parametrisation): ")    
    U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
    #print(U_)

    if (UF.shape[1] > model.S):
        index_ = int(model.S*((UF.shape[1]/model.S -1))/2)
    else:   
        index_ = 0
        
    CentralFloquetEVec_RBM[i,:,0] = UF[:,index_  ].numpy()
    CentralFloquetEVec_RBM[i,:,1] = UF[:,index_+1].numpy()
    CentralFloquetE_RBM[i,0] = tf.math.real(U_[index_     , index_    ]).numpy()
    CentralFloquetE_RBM[i,1] = tf.math.real(U_[index_ + 1 , index_ + 1]).numpy() 

    plt.plot(np.abs(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2]))#print(U_)
    plt.show()
    plt.plot(np.abs(model.U_Floquet[:,:]))#print(U_)
    plt.show()

    #script_result  = [model.trainable_variables,loss_value.numpy()] 
    #script_result = [np.real(UF.numpy()),np.imag(UF.numpy()),loss_value.numpy()]
    script_result = [np.real(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2].numpy()),np.imag(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2].numpy()),loss_value.numpy()]
    
    Model.save_RBM(file,Script_Results=script_result)
    #print("Hamiltonian in the dressed basis (Floquet): ")    
    #U_ = tf.math.real(tf.transpose(tf.math.conj(model.U_Floquet))@model.H_TLS@model.U_Floquet)
    #print(U_)
    
    
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

    #trained_parameters_0 = [model.W_n,model.b_n,model.c_n,model.W_ph,model.b_ph,model.c_ph]


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, 
                                         epsilon=1e-07, amsgrad=False,name='Adam')
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
    loss_list = tf.Variable([],dtype=tf.float64)

    CentralFloquetE_RBM2    = np.zeros([N,2],      dtype = np.float64)
    CentralFloquetEVec_RBM2 = np.zeros([N,H.dim,2],dtype = np.complex128)


    f = 1.0
    loss_list = tf.Variable([],dtype=tf.float64)
    #for i in [16]:#range(1):

    delta = 1.7
    Omega = 10.0*(i)/N 
    phase = np.arctan(1.0)/4.1341341

    #trained_parameters = model.trainable_variables
    #model = Model.RBM_Model(delta,Omega,phase,trained_parameters_0)
    #model = Model.RBM_Model(delta,Omega,phase)
    loss_value, grads = Model.grad_fun(model,Model.loss)
    print("Initial loss value (Floquet): ",loss_value.numpy())    
    
    UF = Model.Unitary_Matrix(model) 

    #### initial guess    
    plt.plot(np.abs(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2]))#print(U_)
    plt.show()
    ### inintial error
    U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
    plt.imshow(np.abs(U_.numpy()))
    plt.show()

    #init_parameters = [f*model.W_n,f*model.b_n,f*model.c_n,f*model.W_ph,f*model.b_ph,f*model.c_ph]
    
    #test_parameters_ = init_parameters + trained_parameters_0 
    
    #model = Model.RBM_Model(delta,Omega,phase,test_parameters_)
    #UF_0  = Model.Unitary_Matrix(model) 
    
    CentralFloquetE[i,0]      = tf.math.real(model.E_Floquet[0]).numpy()
    CentralFloquetE[i,1]      = tf.math.real(model.E_Floquet[1]).numpy() 
    CentralFloquetEVec[i,:,0] = model.U_Floquet[:,0].numpy()
    CentralFloquetEVec[i,:,1] = model.U_Floquet[:,1].numpy()


    N_training = N_training_Matrix

    for i_ in range(N_training):
        loss_value, grads = Model.grad_fun(model,Model.loss)
        loss_list = tf.concat([loss_list,[loss_value.numpy()]],axis=0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if(np.mod(i_,128)):
            script_result = [np.real(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2].numpy()),np.imag(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2].numpy()),loss_value.numpy()]
            #script_result = [model.trainable_variables[0].numpy(),loss_value.numpy()] 
            Model.save_RBM(file,Script_Results=script_result)
        if(i_ == (N_training-1)):
            UF = Model.Unitary_Matrix(model) 
            ##        print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))
            U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
            #print(U_)
            print("Final loss value (Floquet): ",loss_value.numpy())    
    
    if (UF.shape[1] > model.S):
        index_ = int(model.S*((UF.shape[1]/model.S -1))/2)
    else:   
        index_ = 0

    UF = Model.Unitary_Matrix(model) 
    plt.plot(np.abs(UF[:,model.N_Floquet_UF*model.S:model.N_Floquet_UF*model.S+2]))#print(U_)
    plt.show()
    plt.plot(np.abs(model.U_Floquet[:,:]))#print(U_)
    plt.show()
    ##        print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))
    U_ = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
    plt.imshow(np.abs(U_.numpy()))
    plt.show()


    CentralFloquetE_RBM2[i,0]      = tf.math.real(U_[index_   , index_  ]).numpy()
    CentralFloquetE_RBM2[i,1]      = tf.math.real(U_[index_+1 , index_+1]).numpy() 
    CentralFloquetEVec_RBM2[i,:,0] = UF[:,index_].numpy()
    CentralFloquetEVec_RBM2[i,:,1] = UF[:,index_+1].numpy()
    
    plt.plot(loss_list)
    plt.show()
    
    


#mpl.pyplot.plot(loss_list)
    
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

#trained_parameters = [model.W_n,model.b_n,model.c_n,model.W_ph,model.b_ph,model.c_ph]

#%%


#model_Floquet = Model.RBM_Model(delta,Omega,phase,trained_parameters_0)
#U_F = tf.math.real(tf.transpose(tf.math.conj(model_Floquet.UF))@model.H_TLS@model_Floquet.UF)
#print(Model.loss(model_Floquet))
#plt.plot(np.abs(model_Floquet.UF[:,10]))#print(U_)
#plt.show()

#model_Trained = Model.RBM_Model(delta,Omega,phase,trained_parameters)
#U_T = tf.math.real(tf.transpose(tf.math.conj(model_Trained.UF))@model.H_TLS@model_Trained.UF)
#print(Model.loss(model_Trained))
#plt.plot(np.abs(model_Trained.UF[:,10]))
#plt.show()


#model_Random  = Model.RBM_Model(delta,Omega,phase)
#U_R = tf.math.real(tf.transpose(tf.math.conj(model_Random.UF))@model.H_TLS@model_Random.UF)
#print(Model.loss(model_Random))
#plt.plot(np.abs(model_Random.UF[:,12]))
#plt.show()
#print(U_)

















