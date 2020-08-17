# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy
import pandas as pd
import os
import sys
import time
from tensorflow import keras
import networkx as nx

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class ConvolutionLayer(keras.layers.Layer):
    '''
    Convulution Layer.
    input_dim, output_dim, adj, dropout rate, activation function
    return output_dim vector.
    '''
    def __init__(self,input_dim, output_dim, adj, dropout=0., activation='relu', **kwargs):
        super(ConvolutionLayer,self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.activation = keras.layers.Activation(activation)
        self.dropout = dropout

    def build(self, input_shape):
        self.kernal = self.add_weight(name='kernel',
                                        shape = (self.input_dim, self.output_dim),
                                        initializer = 'glorot_uniform',
                                        trainable = True)

    def call(self, x):
        x = tf.matmul(x, self.kernal)
        x = tf.sparse.sparse_dense_matmul(self.adj,x)
        output = self.activation(x)
        return output

class ConvolutionSparseLayer(keras.layers.Layer):
    '''
    Sparse Convulution Layer.
    input_dim, output_dim, adj, dropout rate, features that are nonzero, activation function
    return output_dim vector.
    '''
    def __init__(self,input_dim, output_dim, adj, features_nonzero, dropout=1., activation='relu', **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.adj = adj
        self.activation = keras.layers.Activation(activation)
        self.dropout = dropout
        self.features_nonzero = features_nonzero
        super(ConvolutionSparseLayer,self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernal = self.add_weight(name='kernel',
                                        shape = (input_shape[1], self.output_dim),
                                        initializer = 'glorot_uniform',
                                        trainable = True)

    def call(self, x):
        x = tf.sparse.sparse_dense_matmul(x, self.kernal)
        self.adj= tf.sparse.reorder(self.adj)
        x = tf.sparse.sparse_dense_matmul(self.adj,x)
        output = self.activation(x)
        return output

class WeightedConcateLayer(keras.layers.Layer):
    '''
    Aggregation Layer
    '''
    def __init__(self,input_dim, activation, output_dim=1, **kwargs):
        super(WeightedConcateLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = keras.layers.Activation(activation)

    def build(self, input_shape):
        self.kernal = self.add_weight(name='kernel',
                                        shape = (self.input_dim, self.output_dim),
                                        initializer = tf.constant_initializer(1.),
                                        trainable = True)

    def call(self, embs):
        # feed forward computation
        weighted_embs = {}
        for index, key in enumerate(embs.keys()):
            weight = self.kernal[index]
            emb = weight * embs[key]
            weighted_embs.update({key : emb}) 
        output = tf.concat([weighted_embs[V] for V in list(weighted_embs.keys())], 1)
        output = self.activation(output)
        return output

class InnerProductDecoder(keras.layers.Layer):
    """
    Inner-Product Decoder, reconstruct the adjacency matrix.
    """
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim

    def build(self, input_shape):
        super(InnerProductDecoder, self).build(self.input_dim)    

    def call(self, inputs):
        '''
        return the reconstructed adjacent matrix
        '''
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        output = self.act(x)
        return output