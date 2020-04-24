#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:45:37 2020

@author: German Sinuco
         gsinuco@gmail.com
         
Part of the project RBM parametrisation of Floquet systems.


Definition of the model: Definition of the Hamiltonian components
                         Definition of the Floquet Hamiltonian
                         RBM parametrisation of the micromotion operator
                         Definition of the loss function and its gradient
                         definition of the training sequence
                         
"""
import tensorflow as tf
import numpy as np
import math as m

import matrixmanipulation as mp

class Hamiltonian():
  def __init__(self,delta=0.0,Omega=0.1,phase=0.0):

    self.spin    = True    
    self.omega_0 = 1.00

    self.delta   = delta # detuning
    self.omega   = self.delta + self.omega_0
    self.Omega   = Omega # 1/2 coupling
    self.phase   = phase # the phase in cos(omega t + phase)

    self.S   = 2  # spin 3.2. Hilbert space dimension
    #self.S   = 2   # spin 1/2. Hilbert space dimension
    self.N   = 16   # Number of positive Floquet manifolds
    self.dim = self.S*(2*self.N+1) # Dimension of the extended floquet space
    zero_ = tf.constant(0.0,dtype=tf.float64)
    one_  = tf.constant(1.0,dtype=tf.float64)
    j_ = tf.constant(tf.complex(zero_,one_),dtype=tf.complex128)
    
    if self.S == 2:
        # spin 1/2 
        self.Identity   =     tf.constant([[1.0,0.0],[ 0.0, 1.0]],dtype = tf.complex128)
        self.Sx         = 0.5*tf.constant([[0.0,1.0],[ 1.0, 0.0]],dtype = tf.complex128)
        self.Sy         = j_*0.5*tf.constant([[0.0,1.0],[-1.0, 0.0]],dtype = tf.complex128)
        self.Sz         = 0.5*tf.constant([[1.0,0.0],[ 0.0,-1.0]],dtype = tf.complex128)
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

#1st April: to do build the hamiltonian matrix
#micromotion operator built from central section, but the symmetry of the unitary matrix does not look right following the Gram-Schmidt procedure
        
    self.H_TLS = FloquetHamiltonian(self)  # Floquet Hamiltonian
    

    E_Floquet, U_Floquet = tf.linalg.eigh(self.H_TLS)    
    self.E_Floquet = E_Floquet[int(self.dim/2 - 1):int(self.dim/2 - 1)+self.S]
    self.U_Floquet = U_Floquet[:,int(self.dim/2 - 1):int(self.dim/2 - 1)+self.S]

    self.E_RWA = U_RWA(self)[0]
    self.U_RWA = U_RWA(self)[1][:,self.S:2*self.S]
    

def U_RWA(H):
    
      zero_ = tf.constant(0.0,dtype=tf.float64)
      one_  = tf.constant(1.0,dtype=tf.float64)
      j_ = tf.constant(tf.complex(zero_,one_),dtype=tf.complex128)
      H_FieldColumn = H.Szero
      for i in range(-1,-2):
            H_FieldColumn = tf.concat([H_FieldColumn,H.Szero],axis=0)

      if(1>1): H_FieldColumn = tf.concat([H_FieldColumn,H.Szero],axis=0)   
      H_FieldColumn = tf.concat([H_FieldColumn,H.omega*H.Identity],axis=0)
      H_FieldColumn = tf.concat([H_FieldColumn,H.Szero],axis=0)

      for i in range(1,1):
            H_FieldColumn = tf.concat([H_FieldColumn,H.Szero],axis=0)

      H_FloquetColumn = tf.exp(j_*H.phase)*H.Omega*H.Sx
      H_FloquetColumn = tf.concat([H_FloquetColumn,H.omega_0*H.Sz],axis=0)
      H_FloquetColumn = tf.concat([H_FloquetColumn,H.Omega  *H.Sx*tf.exp(-j_*H.phase)],axis=0)
        # Window to modulate  tf.roll    
      window = tf.concat([tf.ones([H.S*2,H.S],tf.complex128),tf.zeros([H.S,H.S],tf.complex128)],axis=0)
    
      basis = tf.math.multiply(tf.roll(H_FloquetColumn,shift=int(-H.S),axis=0),window)    
      basis = tf.math.add(basis,-tf.roll(H_FieldColumn,shift=int(-H.S),axis=0))

      counter = 1
      for i in range(0,2):#vectors.get_shape()[0].value):
          if (i <= 0):
              window = tf.concat([tf.ones([H.S*(2+counter),H.S],tf.complex128),tf.zeros([H.S*(1-counter),H.S],tf.complex128)],axis=0)
          if (i>0):   
              window = tf.concat([tf.zeros([H.S*(1 + counter),H.S],tf.complex128),tf.ones([H.S*(2-counter),H.S],tf.complex128)],axis=0)
            
          H_FC_ = tf.math.multiply(tf.roll(H_FloquetColumn,shift = int(i*H.S), axis=0),window)        
          H_FC_ = tf.math.add(H_FC_,i*tf.roll(H_FieldColumn,shift=int(i*H.S),axis=0))

          basis = tf.concat([basis, H_FC_],axis=1)
          counter = counter + 1
          if (i == 0):    
              counter = 0
        
      #print(basis)
      E,U = tf.linalg.eigh(basis)
      #print(E)
      #print(V)
      return E,U
        
        
        

class RBM_Model(Hamiltonian,object):
  
  def __init__(self,delta=0.0,Omega=0.1,phase=0.0,BiasWeights=0):
  #def __init__(self,delta=0.0,Omega=0.1,phase=0.0):
      
    pi = tf.atan(1.0)
    
    Hamiltonian.__init__(self,delta,Omega,phase)  

    # Initialize the spin value and number of floquet channels
    self.hidden_n  = 16 # hidden neurons
    self.hidden_ph = 32  # hidden neurons


    self.N_Floquet_UF = self.N/2 # Number of Floquet manifolds used to build the micromotion operator
    
    if BiasWeights:
    # Declaring training variables
    # Training parameters defining the norm 
    #self.W_n   = tf.Variable(tf.zeros([self.hidden_n,self.S*self.dim,3],dtype=tf.float64)
    #                                                  ,trainable=True) 
        self.W_n  = BiasWeights[0]
        self.b_n  = BiasWeights[1]
        self.c_n  = BiasWeights[2]
        self.W_ph = BiasWeights[3]
        self.b_ph = BiasWeights[4]
        self.c_ph = BiasWeights[5]

    else:
        
        if (self.spin == False):
            # When the Hamiltonian does not display any symmetry, the coupling tensor has the form:
            self.W_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n,self.S*self.dim,3],
                                                                 seed=[1,2],dtype=tf.float64,
                                                                 minval=-1.0,maxval=1.0),trainable=True) 
            self.b_n   = tf.Variable(tf.random.stateless_uniform([self.S*self.dim,3], 
                                                                 seed=[2,1],dtype=tf.float64,
                                                                 minval=-1.0,maxval=1.0),trainable=True)
            
            # 17th April 2020
            # in spin systems, the magnitude of the coeficients satisfy P(x) = P(-x)
            # and we reqiure less parameters. This also modifies the way we build the transformation    matrix
            # def Unitary_Matrix(model)
        else:
            self.W_n   =    tf.Variable(tf.random.stateless_uniform([self.hidden_n,int(0.5*self.S*self.dim),3],
                                                           seed=[1,2],dtype=tf.float64,
                                                           minval=-1.0,maxval=1.0),trainable=True) 
            self.b_n   = tf.Variable(tf.random.stateless_uniform([int(0.5*self.S*self.dim),3], 
                                                           seed=[2,1],dtype=tf.float64,
                                                          minval=-1.0,maxval=1.0),trainable=True)
            #self.W_n   = tf.Variable(tf.random.uniform([self.hidden_n,int(0.5*self.S*self.dim),3],
            #                                                dtype=tf.float64,
            #                                                minval=-1.0,maxval=1.0),trainable=True) 
            #self.b_n   = tf.Variable(tf.random.uniform([int(0.5*self.S*self.dim),3], 
            #                                                dtype=tf.float64,
            #                                                minval=-1.0,maxval=1.0),trainable=True)
                
            ### ENHANCING THE WEIGHT OF THE CENTRAL MANIFOLD
            #for i in range(self.S):
            #    index_aux = self.S*self.N + i*(self.S*(2*self.N+1))
            #    self.W_n[:,index_aux:index_aux+self.S,:].assign(tf.random.stateless_uniform([self.hidden_n,self.S,3],
            #                                               seed=[5,2],dtype=tf.float64,
            #                                               minval=-2.0,maxval=2.0))
            #    index_aux = self.S*self.N + i*(self.S*(2*self.N+1))
            #    self.b_n[index_aux:index_aux+self.S,:].assign(tf.random.stateless_uniform([self.S,  3],
            #                                               seed=[2,7],dtype=tf.float64,
            #                                               minval=-2.0,maxval=2.0))
                            
            #self.c_n   = tf.Variable(tf.zeros([self.hidden_n],dtype=tf.float64)
            #                                                  ,trainable=True)
            self.c_n   = tf.Variable(tf.random.stateless_uniform([self.hidden_n],                    
                                                              seed=[1,3],dtype=tf.float64,
                                                              minval=-1.0,maxval=1.0),trainable=True)
            #self.c_n   = tf.Variable(tf.random.uniform([self.hidden_n],                    
            #                                           dtype=tf.float64,
            #                                           minval=-1.0,maxval=1.0),trainable=True)
    
            # Training parameters defining the phase
            #self.W_ph   = tf.Variable(tf.ones([self.hidden_n,self.S*self.dim,3],dtype=tf.float64)
            #                                                  ,trainable=True) 
            self.W_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph,self.S*self.dim,3],
                                                               seed=[3,1],dtype=tf.float64,
                                                               minval=-10.0,maxval=10.0),trainable=True) 
            #self.W_ph   = tf.Variable(tf.random.uniform([self.hidden_ph,self.S*self.dim,3],
            #                                           dtype=tf.float64,
            #                                           minval=-10.0,maxval=10.0),trainable=True) 
            #self.b_ph   = tf.Variable(tf.ones([self.S*self.dim,3],dtype=tf.float64)
            #                                                   ,trainable=True)
            self.b_ph   = tf.Variable(tf.random.stateless_uniform([self.S*self.dim,3], 
                                                               seed=[1,1],dtype=tf.float64,
                                                               minval=-10.0,maxval=10.0),trainable=True)
            #self.b_ph   = tf.Variable(tf.random.uniform([self.S*self.dim,3], 
            #                                           dtype=tf.float64,
            #                                           minval=-10.0,maxval=10.0),trainable=True)
            #self.c_ph   = tf.Variable(tf.ones([self.hidden_n],dtype=tf.float64)
            #                                                  ,trainable=True)
            self.c_ph   = tf.Variable(tf.random.stateless_uniform([self.hidden_ph],                    
                                                               seed=[1,1],dtype=tf.float64,
                                                               minval=-10.0,maxval=10.0),trainable=True)
            #self.c_ph   = tf.Variable(tf.random.uniform([self.hidden_ph],                    
            #                                           dtype=tf.float64,
            #                                           minval=-10.0,maxval=10.0),trainable=True)
        
    # defining the labels of the input layer, which are the components of the UF matrix
    self.x = tf.Variable([[0.0,0.0,0.0]],dtype=tf.float64)
    counter = 0
    self.count = counter
    for j in range(1,self.S+1):
        for l in range(-self.N,self.N+1):
            for i in range(1,self.S+1):        
                if(self.S==4):
                    if(l!=0): y = [[-i+2.5,j-2.5,1.0/l]]
                    if(l==0): y = [[-i+2.5,j-2.5,1.0]]
                    #if(l!=0): y = [[i-2.5,0,1.0/l]]
                    #if(l==0): y = [[i-2.5,0,l]]

                if(self.S==2):
                    if(l!=0): y = [[-i+1.5,j-1.5,1.0/l]]
                    if(l==0): y = [[-i+1.5,j-1.5,1.0]]
                    #if(l!=0): y = [[i-1.5,0,1.0/l]]
                    #if(l==0): y = [[i-1.5,0,l]]
                self.x = tf.concat([self.x, y], 0) 
                counter +=1
                self.count = counter

    self.UF = Unitary_Matrix(self)
    
    #tf.random.stateless_uniform([self.S*(self.N-1),self.S],                    
    #                                                   seed=[1,3],dtype=tf.flo,
    #                                                   minval=-10.0,maxval=10.0),trainable=True)
    
    zeros_aux  = tf.zeros([self.S*(self.N-1),self.S],dtype=tf.complex128)    
    self.U_RWA = tf.concat([zeros_aux,self.U_RWA,zeros_aux],axis=0)
            
    self.trainable_variables = [self.W_n,self.b_n,self.c_n,self.W_ph,self.b_ph,self.c_ph]
    #self.trainable_variables = [self.W_n,self.b_n,self.c_n]
        
  def getH(self):
    return self.H_TLS
    
  def __call__(self):     
    return self.H_TLS


def FloquetHamiltonian(model):
    zero_ = tf.constant(0.0,dtype=tf.float64)
    one_  = tf.constant(1.0,dtype=tf.float64)
    j_ = tf.constant(tf.complex(zero_,one_),dtype=tf.complex128)
    
    # Definition of the field quantum shift
    if(model.N > 0):
        H_FieldColumn = model.Szero
        for i in range(-model.N,-2):
            H_FieldColumn = tf.concat([H_FieldColumn,model.Szero],axis=0)

        if(model.N>1): H_FieldColumn = tf.concat([H_FieldColumn,model.Szero],axis=0)   
        H_FieldColumn = tf.concat([H_FieldColumn,model.omega*model.Identity],axis=0)
        H_FieldColumn = tf.concat([H_FieldColumn,model.Szero],axis=0)

        for i in range(1,model.N):
            H_FieldColumn = tf.concat([H_FieldColumn,model.Szero],axis=0)

    #Definition of the Floquet central column
        
        if(model.N>1): 
            H_FloquetColumn = model.Szero
            for i in range(-model.N,-2):
                H_FloquetColumn = tf.concat([H_FloquetColumn,model.Szero],axis=0)

            H_FloquetColumn = tf.concat([H_FloquetColumn,tf.exp(j_*model.phase)*model.Omega  *model.Sx],axis=0)
            H_FloquetColumn = tf.concat([H_FloquetColumn,model.omega_0*model.Sz],axis=0)
            H_FloquetColumn = tf.concat([H_FloquetColumn,tf.exp(-j_*model.phase)*model.Omega  *model.Sx],axis=0)
        if(model.N==1):
            H_FloquetColumn = tf.exp(j_*model.phase)*model.Omega*model.Sx
            H_FloquetColumn = tf.concat([H_FloquetColumn,model.omega_0*model.Sz],axis=0)
            H_FloquetColumn = tf.concat([H_FloquetColumn,tf.exp(-j_*model.phase)*model.Omega  *model.Sx],axis=0)


        for i in range(1,model.N):
            H_FloquetColumn = tf.concat([H_FloquetColumn,model.Szero],axis=0)
    
        # Window to modulate  tf.roll    
        window = tf.concat([tf.ones([model.S*(model.N+1),model.S],tf.complex128),tf.zeros([model.S*(model.N),model.S],tf.complex128)],axis=0)
    
        basis = tf.math.multiply(tf.roll(H_FloquetColumn,shift=int(-model.N*model.S),axis=0),window)    
        basis = tf.math.add(basis,-model.N*tf.roll(H_FieldColumn,shift=int(-model.N*model.S),axis=0))

        counter = 1
        for i in range(-model.N+1,model.N+1):#vectors.get_shape()[0].value):
            if (i <= 0):
                window = tf.concat([tf.ones([model.S*(model.N+1 + counter),model.S],tf.complex128),tf.zeros([model.S*(model.N-counter),model.S],tf.complex128)],axis=0)
            if (i>0):   
                window = tf.concat([tf.zeros([model.S*(model.N + counter),model.S],tf.complex128),tf.ones([model.S*(model.N+1-counter),model.S],tf.complex128)],axis=0)
            
            H_FC_ = tf.math.multiply(tf.roll(H_FloquetColumn,shift = int(i*model.S), axis=0),window)        
            H_FC_ = tf.math.add(H_FC_,i*tf.roll(H_FieldColumn,shift=int(i*model.S),axis=0))

            basis = tf.concat([basis, H_FC_],axis=1)
            counter = counter + 1
            if (i == 0):    
                counter = 0
    if(model.N == 0):
        basis = model.omega_0*model.Sz+model.Omega*model.Sx

    return basis
    
    

def Unitary_Matrix(model): 

    counter = model.count 

    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer, of the central floquet manifold    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    UF_n     = Unitary_Matrix_Norm(model) 
    # 2. phase 
    UF_ph = Unitary_Matrix_Phase(model)     
    
    UF_cos = tf.cos(UF_ph/2.0)
    UF_sin = tf.sin(UF_ph/2.0)    
    UF =tf.complex(UF_n*UF_cos,UF_n*UF_sin)
    UF = mp.normalisation(UF)
    UF = mp.tf_gram_schmidt(UF)

    # G-S algorithm 
    # reported in  https://stackoverflow.com/questions/48119473/gram-schmidt-orthogonalization-in-pure-tensorflow-performance-for-iterative-sol. 
    # To do: incorparate a basis rotation in the training loop

# ===== manipulation of the central floquet UF to form a big Floquet UF
    # 1st of March 2020. Task: REVISE NORMALISATION AND GRAM-SCHMIDT PROCEDURE FOR COMPLEX VECTORS
    # 5th of March 2020. Normalisation done by hand: OK. Now I am using the G-S algorithm 
    #                    reported in  https://stackoverflow.com/questions/48119473/gram-schmidt-orthogonalization-in-pure-tensorflow-performance-for-iterative-sol. 
    #                    Task: incorparate a basis rotation in the training loop
    # 23rd April 2020 To find the Floquet spectrum, I will try using more than a single Floquet manifold.



    window = tf.concat([tf.ones([model.S*(model.N+1),model.S],tf.complex128),tf.zeros([model.S*(model.N),model.S],tf.complex128)],axis=0)
    
    basis = tf.math.multiply(tf.roll(UF,shift=int(-model.N*model.S),axis=0),window)    
  
    counter = 1
    for i in range(-model.N+1,model.N+1):#vectors.get_shape()[0].value):        
        if (i <= 0):
            window = tf.concat([tf.ones([model.S*(model.N+1 + counter),model.S],tf.complex128),tf.zeros([model.S*(model.N-counter),model.S],tf.complex128)],axis=0)
        if (i>0):   
            window = tf.concat([tf.zeros([model.S*(model.N + counter),model.S],tf.complex128),tf.ones([model.S*(model.N+1-counter),model.S],tf.complex128)],axis=0)
            
        UF_ = tf.math.multiply(tf.roll(UF,shift = int(i*model.S), axis=0),window)        
        # 1st April. This roll is periodic, so it's now useful        
        basis = tf.concat([basis, UF_],axis=1)
        counter = counter + 1
        if (i == 0):    
            counter = 0

 
    basis = mp.normalisation(basis)
    ###### WARNING! AVOID TO ORTHOGONALISE UF THAT INCLUDE MORE THAN THE CENTRAL MANIFOLD ###
    ####### TO DO: WRITING A SPECIAL ORTHONORMALISATION ROUTINE THAT DOES NOT MODIFY THE CENTRAL
    ####### MANIFOLD
    #basis = mp.tf_gram_schmidt(basis)    

    return basis
    #return UF


def Unitary_Matrix_Norm(model): 

    counter = model.count 

    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    # WHEN THE HAMILTONIAN DOES NOT HAVE ANY SYMMETRY, USE THE FOLLOWING
    if (model.spin == False):
        WX_n = [tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_n[0]),1)+model.c_n[0]]
        for j in range(1,model.hidden_n):
            y = tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_n[j]),1)+model.c_n[j]
            WX_n = tf.concat([WX_n, [y]], 0) 
        
        UF_aux = tf.sqrt(tf.abs(tf.multiply(tf.reduce_prod(tf.math.cosh(WX_n),0),tf.exp(
                                            tf.transpose(tf.reduce_sum(tf.multiply(
                                                model.x[1:counter+1],model.b_n),1))))))

    else:        
    # 17 APRIL 2020: FOR SPIN SYSTEMS
    #we only have to evaluate half of the reduction, because of the symmetry of the spectrum
        WX_n = [tf.reduce_sum(tf.multiply(model.x[1:int(counter/2)+1],model.W_n[0]),1)+model.c_n[0]]
        for j in range(1,model.hidden_n):
            y = tf.reduce_sum(tf.multiply(model.x[1:int(counter/2)+1],model.W_n[j]),1)+model.c_n[j]
            WX_n = tf.concat([WX_n, [y]], 0) 
                    
        UF_aux = tf.sqrt(tf.abs(tf.multiply(tf.reduce_prod(tf.math.cosh(WX_n),0),tf.exp(
                                        tf.transpose(tf.reduce_sum(tf.multiply(
                                                model.x[1:int(0.5*counter)+1],model.b_n),1))))))
        
        UF_aux = tf.concat([UF_aux,tf.reverse(UF_aux,[0])],0)
        
        
    UF_n = tf.reshape(UF_aux,[model.dim,model.S])
    
    return UF_n


def Unitary_Matrix_Phase(model): 

    counter = model.count 
    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 2. Phase 
    WX_ph = [tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_ph[0]),1)+model.c_ph[0]]
    for j in range(1,model.hidden_ph):
        y = tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.W_ph[j]),1)+model.c_ph[j]
        WX_ph = tf.concat([WX_ph, [y]], 0) 
        
    UF_aux = tf.multiply(tf.reduce_prod(tf.math.cosh(WX_ph),0),tf.exp(
            tf.transpose(tf.reduce_sum(tf.multiply(model.x[1:counter+1],model.b_ph),1))))
    UF_ph = tf.reshape(tf.math.log(UF_aux),[model.dim,model.S])
    
    return UF_ph



def train(model,learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model)
  dU = t.gradient(current_loss, model.trainable_variables)
  model.UF.assign_sub(learning_rate*dU)


def train_with_loss(model,learning_rate,loss_fun):
  with tf.GradientTape() as t:
    current_loss = loss_fun(model)
  dU = t.gradient(current_loss, model.trainable_variables)
  model.UF.assign_sub(learning_rate*dU)


# 3e. Loss function := Use U^dagger H U, sum over the columns, take the difference with the diagonal, 
#     the loss function is the summ of the square of these differences. 

def loss(model):
  # define the loss function explicitly including the training variables: model.W, model.b
  # model.UF is a function of self.W,self.b,self.c
    #UF    = tf.Variable(np.zeros((model.dim*model.dim), dtype=np.complex64),trainable = False)  # ext. micromotion operator
  
    #UF = tf.Variable(np.zeros((model.dim,model.dim),dtype=np.complex64)) 
    #a  = np.zeros((model.dim,model.dim),dtype=np.float32)
    #counter = model.count 

    #Building of the marginal probability of the RBM using the training parameters and labels of the input layer    
    #P(x)(b,c,W) = exp(bji . x) Prod_l=1^M 2 x cosh(c_l + W_{x,l} . x)
    # 1. Amplitude (norm)
    UF = Unitary_Matrix(model)
    #print(tf.abs(tf.matmul(UF,UF,adjoint_a=True)))
    
    U_      = tf.transpose(tf.math.conj(UF))@model.H_TLS@UF
    #print(U_)
    U_diag  = tf.abs(tf.linalg.tensor_diag_part(U_))  
    dotProd = tf.math.reduce_sum(abs(U_),axis=1,)    
    #residual = tf.math.sqrt(tf.math.reduce_sum(tf.pow((U_diag-dotProd),2),0))
    #Residual: defined to minimise the difference between U_ and a diagonal form  + 
    #          
    #residual = tf.math.reduce_sum(tf.abs(U_diag-dotProd),0)
    residual = tf.math.reduce_sum(tf.math.sqrt(tf.abs(U_diag-dotProd)),0) + 1.0*tf.math.abs(tf.linalg.trace(U_))
    
    #tf.math.sqrt(tf.math.reduce_sum(tf.abs(U_diag-dotProd),0)) #+ tf.math.reduce_sum(tf.abs(U_diag),0)
    
    #U_ = tf.abs(tf.transpose(tf.math.conj(UF))@UF)
    #print(U_)
    #U_diag = tf.linalg.tensor_diag_part(U_)  

    #dotProd = tf.math.reduce_sum(abs(U_),axis=1)    

    #residual_unitary = tf.pow(tf.math.reduce_sum(dotProd,0) - model.dim,2.0)
    
    #residual += 7.0*residual_unitary

    return residual


def loss_RWA(model):
    #print(model.U_RWA)
    #print(model.UF)
    k        = tf.keras.losses.KLDivergence()
    UF       = Unitary_Matrix(model)    
    U_Target = model.U_RWA
    
    #for i in range(0,N)
    #U_basis = tf.random.stateless_uniform(UF.shape,seed=[2,1],dtype=tf.float64,minval=-1.0,maxval=1.0)
    #U_basis = tf.linalg.svd(U_basis)[1]
    
    #loss_  = k(tf.math.square(tf.abs(model.U_RWA[:,0])),tf.math.square(tf.abs(UF[:,0])))
    #basis_ = tf.expand_dims(loss_,0)
    #for i in range(UF.shape[1]-1):
    #    basis_ =  tf.concat([basis_,tf.expand_dims(tf.math.square(tf.abs(model.U_RWA[:,i+1])),tf.math.square(tf.abs(UF[:,i+1])),1)],axis=0)
    if(UF.shape[1]>model.S): 
        index_ = int(model.S*((UF.shape[1]/model.S -1))/2)#np.int(((model.S+1)*model.N))
    else: 
        index_ = 0
            
    loss = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF[:,index_])))
    for i in range(U_Target.shape[1]-1):
        ### WARNING ! DOES TF KEEP TRACK OF THE GRADIENTS WHEN THE LOSS IS ITERATIVE? =####
        ###           OR SHOULD WE DEFINE AN ARRAY OF LOSSES AND THEN REDUCE THEM?     ####
        loss = loss + k(tf.math.square(tf.abs(U_Target[:,i+1])),tf.math.square(tf.abs(UF[:,index_+i+1])))

    #if(model.S==2):
    #    loss_0 = k(tf.math.square(tf.abs(model.U_RWA[:,0])),tf.math.square(tf.abs(UF[:,0])))
    #    loss_1 = k(tf.math.square(tf.abs(model.U_RWA[:,1])),tf.math.square(tf.abs(UF[:,1])))
    #    loss   = loss_0 + loss_1

    #if(model.S==4):
    #    loss_0 = k(tf.math.square(tf.abs(model.U_RWA[:,0])),tf.math.square(tf.abs(UF[:,0])))
    #    loss_1 = k(tf.math.square(tf.abs(model.U_RWA[:,1])),tf.math.square(tf.abs(UF[:,1])))
    #    loss_2 = k(tf.math.square(tf.abs(model.U_RWA[:,2])),tf.math.square(tf.abs(UF[:,2])))
    #    loss_3 = k(tf.math.square(tf.abs(model.U_RWA[:,3])),tf.math.square(tf.abs(UF[:,3])))
    #    loss   = loss_0 + loss_1 + loss_2 + loss_3
    #print(loss)
    #loss = k(tf.reshape(tf.abs(model.UF),[52,1]),tf.reshape(tf.abs(model.U_RWA),[52,1]))
    #print(loss)
    return loss

    #loss = k([.4, .9, .2],[.5, .9, .2])


def loss_RWA_Phase(model):
    #print(model.U_RWA)
    #print(model.UF)
    k        = tf.keras.losses.KLDivergence()
    UF       = Unitary_Matrix(model)    
    U_Target = model.U_RWA
    UF_      = UF    

    #H_       = tf.math.real(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
    #H_Target = tf.math.real(tf.transpose(tf.math.conj(U_Target))@model.H_TLS@U_Target)
    #loss_H   = tf.math.reduce_sum(tf.abs(H_ - H_Target)) + tf.abs(tf.linalg.trace(H_))
    # the first term of loss_H drives an asymmetry of the eigenvalues
    N = 8 # N random basis
    loss  = 0.0
    for i in range(0,N):

        U_basis_R = tf.random.stateless_uniform(model.H_TLS.shape,seed=[3,6],dtype=tf.float64,minval=-1.0,maxval=1.0)
        U_basis_I = tf.random.stateless_uniform(model.H_TLS.shape,seed=[7,3],dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_R = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_I = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        U_basis   = tf.complex(U_basis_R,U_basis_I)
        U_basis   = tf.linalg.svd(U_basis)[1]
    
        if (N>1):
            U_Target = U_basis@model.U_Floquet
            UF_      = U_basis@UF
        
        if(UF_.shape[1]>model.S): 
            index_ = np.int(((2*model.S+1)*model.N))
        else:   
            index_ = 0
            
        if(UF_.shape[1]>model.S): 
            index_ = int(model.S*((UF.shape[1]/model.S -1))/2)#np.int(((model.S+1)*model.N))
        else: 
            index_ = 0
            
        loss = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,index_])))
        for i in range(U_Target.shape[1]-1):
        ### WARNING ! DOES TF KEEP TRACK OF THE GRADIENTS WHEN THE LOSS IS ITERATIVE? =####
        ###           OR SHOULD WE DEFINE AN ARRAY OF LOSSES AND THEN REDUCE THEM?     ####
            loss = loss + k(tf.math.square(tf.abs(U_Target[:,i+1])),tf.math.square(tf.abs(UF_[:,index_+i+1])))

        #if(model.S==2):
        #   loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #   loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:,1])))
        #   loss   = loss + loss_0 + loss_1
            
        #if(model.S==4):
        #    loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #    loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:, 1])))
        #    loss_2 = k(tf.math.square(tf.abs(U_Target[:,2])),tf.math.square(tf.abs(UF_[:, 2])))
        #    loss_3 = k(tf.math.square(tf.abs(U_Target[:,3])),tf.math.square(tf.abs(UF_[:,3])))
        #    loss   = loss + loss_0 + loss_1 + loss_2 + loss_3
                
    return loss/N


def loss_Floquet(model):
    #print(model.U_RWA)
    #print(model.UF)
    k       = tf.keras.losses.KLDivergence()
    UF      = Unitary_Matrix(model)    
    U_Target = model.U_Floquet
    #UF_      = UF    
    
    #H_       = tf.math.real(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
    #H_Target = tf.math.real(tf.transpose(tf.math.conj(U_Target))@model.H_TLS@U_Target)
    #loss_H   = tf.math.reduce_sum(tf.abs(H_ - H_Target))

    N = 1 # N random basis
    loss  = 0.0
    for i in range(0,N):
        #U_basis_R = tf.random.stateless_uniform(model.H_TLS.shape,seed=[2,1],dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_I = tf.random.stateless_uniform(model.H_TLS.shape,seed=[2,1],dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_R = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_I = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis   = tf.complex(U_basis_R,U_basis_I)
        #U_basis   = tf.linalg.svd(U_basis)[1]
    
        #if (N>1):
        #    U_Target = U_basis@model.U_Floquet
        #    UF_      = U_basis@UF
                
        if(UF.shape[1]>model.S): 
            index_ = int(model.S*((UF.shape[1]/model.S -1))/2)#np.int(((model.S+1)*model.N))
        else: 
            index_ = 0
            
        loss = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF[:,index_])))
        for i in range(U_Target.shape[1]-1):
        ### WARNING ! DOES TF KEEP TRACK OF THE GRADIENTS WHEN THE LOSS IS ITERATIVE? =####
        ###           OR SHOULD WE DEFINE AN ARRAY OF LOSSES AND THEN REDUCE THEM?     ####
            loss = loss + k(tf.math.square(tf.abs(U_Target[:,i+1])),tf.math.square(tf.abs(UF[:,index_+i+1])))

            
        #if(model.S==2):
        #    loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #    loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:,1])))
        #    loss   = loss + loss_0 + loss_1
            
        #if(model.S==4):
        #    loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #    loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:, 1])))
        #    loss_2 = k(tf.math.square(tf.abs(U_Target[:,2])),tf.math.square(tf.abs(UF_[:, 2])))
        #    loss_3 = k(tf.math.square(tf.abs(U_Target[:,3])),tf.math.square(tf.abs(UF_[:,3])))
        #    loss   = loss + loss_0 + loss_1 + loss_2 + loss_3
                
    return loss/N


def loss_Floquet_Phase(model):
    #print(model.U_RWA)
    #print(model.UF)
    k       = tf.keras.losses.KLDivergence()
    UF      = Unitary_Matrix(model)    
    U_Target = model.U_Floquet
    UF_      = UF    

    #H_       = tf.math.real(tf.transpose(tf.math.conj(UF))@model.H_TLS@UF)
    #H_Target = tf.math.real(tf.transpose(tf.math.conj(U_Target))@model.H_TLS@U_Target)
    #loss_H   = tf.math.reduce_sum(tf.abs(H_ - H_Target)) + tf.abs(tf.linalg.trace(H_))
    # the first term of loss_H drives an asymmetry of the eigenvalues
    N = 8 # N random basis
    loss  = 0.0
    for i in range(0,N):

        U_basis_R = tf.random.stateless_uniform(model.H_TLS.shape,seed=[3,6],dtype=tf.float64,minval=-1.0,maxval=1.0)
        U_basis_I = tf.random.stateless_uniform(model.H_TLS.shape,seed=[7,3],dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_R = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        #U_basis_I = tf.random.uniform(model.H_TLS.shape,dtype=tf.float64,minval=-1.0,maxval=1.0)
        U_basis   = tf.complex(U_basis_R,U_basis_I)
        U_basis   = tf.linalg.svd(U_basis)[1]
    
        if (N>1):
            U_Target = U_basis@model.U_Floquet
            UF_      = U_basis@UF

        if(UF.shape[1]>model.S): 
            index_ = int(model.S*((UF.shape[1]/model.S -1))/2)#np.int(((model.S+1)*model.N))
        else: 
            index_ = 0
            
        loss = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,index_])))
        for i in range(U_Target.shape[1]-1):
        ### WARNING ! DOES TF KEEP TRACK OF THE GRADIENTS WHEN THE LOSS IS ITERATIVE? =####
        ###           OR SHOULD WE DEFINE AN ARRAY OF LOSSES AND THEN REDUCE THEM?     ####
            loss = loss + k(tf.math.square(tf.abs(U_Target[:,i+1])),tf.math.square(tf.abs(UF_[:,index_+i+1])))

        #if(model.S==2):
        #    loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #    loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:,1])))
        #    loss   = loss + loss_0 + loss_1
            
        #if(model.S==4):
        #    loss_0 = k(tf.math.square(tf.abs(U_Target[:,0])),tf.math.square(tf.abs(UF_[:,0])))
        #    loss_1 = k(tf.math.square(tf.abs(U_Target[:,1])),tf.math.square(tf.abs(UF_[:, 1])))
        #    loss_2 = k(tf.math.square(tf.abs(U_Target[:,2])),tf.math.square(tf.abs(UF_[:, 2])))
        #    loss_3 = k(tf.math.square(tf.abs(U_Target[:,3])),tf.math.square(tf.abs(UF_[:,3])))
        #    loss   = loss + loss_0 + loss_1 + loss_2 + loss_3
                
    return loss/N



# This is the gradient of the loss function. required for keras optimisers
def grad(model):
  with tf.GradientTape() as tape:
    loss_value = loss(model)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def grad_fun(model,loss_fun):
  with tf.GradientTape() as tape:
    loss_value = loss_fun(model)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)


def grad_Phase(model,loss_fun):
  with tf.GradientTape() as tape:
    loss_value = loss_fun(model)
  return loss_value, tape.gradient(loss_value, model.trainable_variables[3:6])
    


def Rect2Pol(x):
    
    ph  = phase(x)
    rho = np.abs(x)
    return rho,ph
         

def phase(x):
    ph =np.arctan(np.abs(np.imag(x)/np.real(x)))
    #A  = np.real(x)
    #B  = np.imag(x)
    #if((A>0).all() & (B<0).all()):
    #    ph = ph + 2.0*np.pi
    #if((A<0).all() & (B>0).all()):
    #    ph = -ph + np.pi
    #if((A<0).all() & (B<0).all()):
    #    ph =  ph + np.pi
    #if((A==0).all() &(B>0).all()):
    #    ph = 0.0
    #if((A==0).all() &(B<0).all()):        
    #    ph = np.pi
    #if((B==0).all() & (np.sign(A)>0).all()):
    #    ph = 0.0
    #if((B==0).all() & (np.sign(A)<0).all()):
    #    print("me")
    #    ph = np.pi

    return ph
                    