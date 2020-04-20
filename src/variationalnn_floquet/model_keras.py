#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:25:03 2020

@author: german
"""

import model as Model
import numpy as np

import tensorflow as tf
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.utils import to_categorical
seed = 1337
np.random.seed(seed)


class Model_keras(Model.Hamiltonian,object):
  
  def __init__(self):
      pi = tf.atan(1.0)    
      Hamiltonian.__init__(self)  
      
      input_     = tf.keras.layers.Input()
      hidden1_n  = tf.keras.layers.Dense(30,activation="relu")(input_)
      hidden2_n  = tf.keras.layers.Dense(1,activation="relu")(hidden2_n)
      hidden1_ph = tf.keras.layers.Dense(30,activation="relu")(input_
      hidden2_ph = tf.keras.layers.Dense(1,activation="relu")(hidden2_ph)
      concat     = tf.keras.layers.concatenate(hidden2_n,hidden2_ph)
     
      
    