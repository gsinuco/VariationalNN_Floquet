#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:58:35 2020

@author: german
"""

from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass


def NN_model(x,y):
    import tensorflow as tf
    import numpy as np

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.utils import to_categorical
    seed = 1337
    np.random.seed(seed)

    N=x.shape[0]

#    x = np.random.random(N)#tf.Variable(np.random.random(128),dtype =tf.float32)
#    y = np.random.choice([0, 1], size=N)#tf.Variable(np.random.choice([0, 1], size=N),dtype = tf.int32)
    #x =tf.Variable(np.random.random(128),dtype =tf.float32)
    #y =tf.Variable(np.random.choice([0, 1], size=N),dtype =tf.int32)
               
    f = 0.7

    x_train = x[0:int(f*N)]
    y_train = (y[0:int(f*N)] +1)/2 

    x_test = x[0:int((1-f)*N)]
    y_test = (y[0:int((1-f)*N)]+1)/2

    num_classes = 2
    y_train = to_categorical(y_train,num_classes) 
    y_test = to_categorical(y_test,num_classes)

    model = Sequential()  # Instantiate sequential model
    model.add(Dense(256, activation='relu', input_shape=x.shape[1:])) # Add first layer. Make sure to specify input    shape. In this case, the input shape is an array of real number, each one a row of x_train
    model.add(Dropout(0.5)) # Add second layer
    model.add(Dense(128,activation='relu')) # Add third layer
    model.add(Dense(2,activation='softmax')) # Add third layer
    
    
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])        
    
    batch_size =  tf.dtypes.cast(N/16,dtype=tf.int32)
    model.fit(x_train,y_train,batch_size=batch_size,epochs=512,validation_data=(x_test,y_test))
    
    score = model.evaluate(x_test,y_test,verbose=1)
     
    return model, score        
       

#import tensorflow as tf
#import numpy as np


#N =66
#x = tf.random.uniform([N,2],dtype=tf.float32).numpy()
#y = tf.transpose(tf.random.categorical([[0.5,0.5]],N)).numpy()
#model_, score = NN_model(x,y)
