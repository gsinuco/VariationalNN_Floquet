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
    basis = tf.expand_dims(vectors[:,0]/tf.norm(vectors[:,0]),1)
    for i in range(1,vectors.shape[1]):#vectors.get_shape()[0].value):
        v = vectors[:,i]
        # add batch dimension for matmul
        v = tf.expand_dims(v,1) 
        #w = v - tf.matmul(tf.matmul(v, basis, adjoint_a=True), basis)
        w = v - tf.matmul(basis,tf.matmul(v, tf.math.conj(basis),transpose_a=True,),transpose_b=True)
        #w = v - tf.matmul(basis,tf.matmul(v, basis))
        #w = v - tf.matmul(tf.matmul(v, basis, adjoint_a=True), basis)
        
        basis = tf.concat([basis, w/tf.norm(w)],axis=1)
    return basis



