#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:46:07 2020

@author: German Sinuco
         gsinuco@gmail.com
         
Part of the project RBM parametrisation of Floquet systems.


Definition of functions for matrix manipulation: Unitary matrix using the RBM parametrisation
                                                 Normalisation of the column vectors of the matrix
                                                 Gram-Schmidt procedure 
                                                 
"""


import tensorflow as tf
import numpy as np
import math as m


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

