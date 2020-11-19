#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:18:41 2019

@author: DigitalSMD_researchers
"""
from tensorflow import keras 
import numpy as np
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.layers import Conv1D, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda
from tensorflow.keras.models import load_model
import tensorflow as tf
### A Test sample of decomposed convolution, but it can not be trained.
### Lamdba function should be used to wrap non-parameter layer
def GroupConv(feature, filter_num):
    input_sequence = K.expand_dims(feature, 2)
    
    #map_0 = Conv1D(filter_num,1, padding="same",activation="relu", kernel_regularizer=l2(0.01))(input_sequence)
    map_1 = Conv1D(filter_num,2, padding="same",activation="relu", kernel_regularizer=l2(0.01))(input_sequence)
    map_2 = Conv1D(filter_num,3, padding="same",activation="relu", kernel_regularizer=l2(0.01))(input_sequence)
    map_3 = Conv1D(filter_num,4, padding="same",activation="relu", kernel_regularizer=l2(0.01))(input_sequence)
    map_4 = Conv1D(filter_num,5, padding="same",activation="relu", kernel_regularizer=l2(0.01))(input_sequence)
    
    MAP = keras.layers.Concatenate( axis=2)([input_sequence, map_1, map_2, map_3, map_4])
    
    #MAP = Conv1D(10, padding="same", activation = "relu", kernel_regularizer=l2(0.01) )(input_sequence)
    return MAP

### The correct way to implement decomposed convolution
class DecomposedConv(Layer):
        # Arguments
        #     filters: the dimentionality of output space (i.e. the number of filters)
        #     padding: control the length of sequence to be the same or to be changed
        #     activation: non-linear function applied to pre-activated convolutional maps
        #     kernel-regularizer: regularizer function applied to kernel matrix
        
    def __init__(self, filters, out_dims, padding="same", strides=1, dilation_rate=1, activation="relu", kernel_regularizer = l2(0.01), **kwargs):
        self.rank = 1
        self.filters = filters
        self.out_dims = out_dims
        #self.data_format = K.normalize_data_format("channels_last")
        self.data_format = conv_utils.normalize_data_format("channels_last")
        #self.kernel_size0 = conv_utils.normalize_tuple(1, self.rank, "kernel_size0")
        self.kernel_size1 = conv_utils.normalize_tuple(2, self.rank, "kernel_size1")
        self.kernel_size2 = conv_utils.normalize_tuple(3, self.rank, "kernel_size2")
        self.kernel_size3 = conv_utils.normalize_tuple(4, self.rank, "kernel_size3")
        self.kernel_size4 = conv_utils.normalize_tuple(5, self.rank, "kernel_size4")
        self.Kernel_size = conv_utils.normalize_tuple(1, self.rank, "Kernel_size")
        self.padding = conv_utils.normalize_padding(padding)
        self.strides = conv_utils.normalize_tuple(strides, self.rank, "strides")
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank,
                                                        'dilation_rate')
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get("glorot_uniform")
        self.bias_initializer = keras.initializers.get("zeros")
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        
        super(DecomposedConv, self).__init__(**kwargs)
        
    def get_config(self):
        config = {'filters': self.filters,
              'out_dims': self.out_dims}
        base_config = super(DecomposedConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
            
        input_dim = input_shape[channel_axis]
        #kernel_shape0 = self.kernel_size0 + (input_dim, self.filters)
        kernel_shape1 = self.kernel_size1 + (input_dim, self.filters)
        kernel_shape2 = self.kernel_size2 + (input_dim, self.filters)
        kernel_shape3 = self.kernel_size3 + (input_dim, self.filters)
        kernel_shape4 = self.kernel_size4 + (input_dim, self.filters)
        
        
        ## initialize kernels for temporal information
        #self.kernel0 = self.add_weight(shape=kernel_shape0,
        #                              initializer=self.kernel_initializer,
        #                              name='kernel0',
        #                              regularizer=self.kernel_regularizer)
        
        self.kernel1 = self.add_weight(shape=kernel_shape1,
                                      initializer=self.kernel_initializer,
                                      name='kernel1',
                                      regularizer=self.kernel_regularizer)
        
        self.kernel2 = self.add_weight(shape=kernel_shape2,
                                      initializer=self.kernel_initializer,
                                      name='kernel2',
                                      regularizer=self.kernel_regularizer)
        self.kernel3 = self.add_weight(shape=kernel_shape3,
                                      initializer=self.kernel_initializer,
                                      name='kernel3',
                                      regularizer=self.kernel_regularizer)
        self.kernel4 = self.add_weight(shape=kernel_shape4,
                                      initializer=self.kernel_initializer,
                                      name='kernel4',
                                      regularizer=self.kernel_regularizer)
        
        #self.bias0 = self.add_weight(shape=(self.filters,),
        #                                initializer=self.bias_initializer,
        #                                name='bias0')
        self.bias1 = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias1')
        self.bias2 = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias2')
        self.bias3 = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias3')
        self.bias4 = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias4')
        
        ## initialize kernel for summarize various temporal patterns
        self.Kernel_shape = self.Kernel_size + (5*self.filters , self.out_dims)
        self.Kernel = self.add_weight(shape = self.Kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name = "Root_kernel",
                                      regularizer=self.kernel_regularizer)
        self.Bias  = self.add_weight(shape = (self.out_dims,),
                                        initializer=self.bias_initializer,
                                        name = "Root_bias")
        
        
        
        self.built = True
        
        super(DecomposedConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, X):
        
        ## kernel of window 1
        #map_0 = K.conv1d(
        #        X,
        #        self.kernel0,
        #        padding=self.padding,
        #        data_format=self.data_format)
    
        #map_0 = K.bias_add(map_0,
        #                   self.bias0,
        #                   self.data_format)
        
        
        ## kernel of window 2
        map_1 = K.conv1d(
                X,
                self.kernel1,
                padding=self.padding,
                data_format=self.data_format)
    
        map_1 = K.bias_add(map_1,
                           self.bias1,
                           self.data_format)
        ## kernel of window 3
        map_2 = K.conv1d(
                X,
                self.kernel2,
                padding=self.padding,
                data_format=self.data_format)
        
        map_2 = K.bias_add(map_2,
                           self.bias2,
                           self.data_format)
        ## kernel of window 4
        map_3 = K.conv1d(
                X,
                self.kernel3,
                padding=self.padding,
                data_format=self.data_format)
        
        map_3 = K.bias_add(map_3,
                           self.bias3,
                           self.data_format)
        
        ## kernel of window 5
        map_4 = K.conv1d(
                X,
                self.kernel4,
                padding=self.padding,
                data_format=self.data_format)
        
        map_4 = K.bias_add(map_4,
                           self.bias4,
                           self.data_format)
        
        MAP = K.concatenate([X, map_1, map_2, map_3, map_4], 2)
        
        MAP = self.activation(MAP)
        
        #### summarizing all the feature maps
        output = K.conv1d(
                 MAP,
                 self.Kernel,
                 padding=self.padding,
                 data_format=self.data_format)
        
        output = K.bias_add(output,
                            self.Bias,
                            self.data_format)
        
        
        return self.activation(output)

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.Kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
        new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.out_dims,)

### function for splitting the input stream into individual fields
def split(X):
    out = []
    for i in range(X.shape[2]):
        out.append(Lambda(lambda x: K.expand_dims(x[:,:,i],2))(X))
    
    return out
### a function to find bn layer for indirect search
def loadmodel(path,custom_layer):
    model = load_model(path, custom_objects=custom_layer)
    
    return model

def model_slim(path,custom_layer, threshold=1):
    
      
    model = path
    bn_weights = model.layers[22].get_weights()[0]
    fields = [0] * 19
    
    """
    for i in range(19):
        if bn_weights[i] > threshold:
            fields[i] = 1
            
    indices = [i for i, x in enumerate(fields) if x == 1]   
    """
    indices = np.argsort(bn_weights)[-5:]  
    indices.sort()          
    # Disassemble layers
    layers = [l for l in model.layers]     


    x = layers[0].output
    x = layers[1](x)
    ###list all the fields
    map1 = layers[2](x[0])
    map2 = layers[3](x[1])
    map3 = layers[4](x[2])
    map4 = layers[5](x[3])
    map5 = layers[6](x[4])
    map6 = layers[7](x[5])
    map7 = layers[8](x[6])
    map8 = layers[9](x[7])
    map9 = layers[10](x[8])
    map10 = layers[11](x[9])
    map11 = layers[12](x[10])
    map12 = layers[13](x[11])
    map13 = layers[14](x[12])
    map14 = layers[15](x[13])
    map15 = layers[16](x[14])
    map16 = layers[17](x[15])
    map17 = layers[18](x[16])
    map18 = layers[19](x[17])
    map19 = layers[20](x[18])
        
    MAP = [map1, map2, map3, map4, map5, map6, map7, map8, map9, map10, map11, map12, map13, map14, map15, map16, map17, map18, map19]
    ###only pick the active fields  
    filtered_fields = [MAP[i] for i in indices]
    MAP = keras.layers.Concatenate(axis=2)(filtered_fields)
    cl = keras.layers.BatchNormalization()
    
    MAP = cl(MAP)
    
    MAP = Conv1D(5, 1,  kernel_regularizer=l1_l2(0.01, 0))(MAP)

    #MAP = layers[24](MAP)
    
    MAP = keras.layers.BatchNormalization()(MAP)
    
    #MAP = layers[25](MAP)
    #MAP = Conv1D(5, 1,  kernel_regularizer=l1_l2(0.01, 0))(MAP)
    
    
    MAP = keras.layers.Concatenate(axis=2)([x[0], MAP])

    MAP = layers[27](MAP)
    #MAP = layers[28](MAP)

    MAP = Dense(1 , kernel_regularizer=l1_l2(0, 0.01))(MAP)
        
    newmodel = keras.Model(inputs = layers[0].input, outputs=MAP)
    newmodel.compile(loss=loss(cl), optimizer="adam")

    print(newmodel.summary())
    
    return newmodel, indices
    

#def loss function with trainable connection parameters
def loss(layer, L1=0.001):
    
    def connection_loss(y_true, y_pred):
        
        return keras.losses.categorical_crossentropy(y_true, y_pred) + L1 * K.sum( K.abs(layer.weights[0]))
    
    return connection_loss
        
      