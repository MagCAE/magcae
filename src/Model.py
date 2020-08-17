# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics.pairwise import rbf_kernel
import scipy
from scipy.spatial import distance
from scipy.sparse import csc_matrix
import pandas as pd
import pickle as pkl
import os
import sys
import time
from tensorflow import keras
import networkx as nx
from layers import ConvolutionLayer,ConvolutionSparseLayer, WeightedConcateLayer,InnerProductDecoder

# Single Attribute view graph convolutional encoder
class GAE(keras.Model):
    '''
    GAE model with 1 sparser convolution layer and 1 convolutuon layer
    initiates:
    input_dim = num of features, 
    hidden1 = num of first hidden layers neurons, 
    embedding_dim = embedding dimension, 
    features_nonezero.
    '''
    def __init__(self, input_dim, features_nonezero, adj,hidden1=32, embedding_dim=16):
        super(GAE, self).__init__()

        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.embedding_dim = embedding_dim
        self.features_nonezero = features_nonezero
        indices= np.array(adj[0])
        values = np.array(adj[1])
        dense_shape = np.array(adj[2])
        sparse_adj = tf.SparseTensor(indices = indices,
                                        values = values,
                                        dense_shape = dense_shape)
        self.adj = tf.cast(sparse_adj, dtype=tf.float32)
        # GAE encoder with 1 sparseconvolution layer, 1 convolution layers
        self.conv1 = ConvolutionSparseLayer(self.input_dim,
                                            self.hidden1,
                                            self.adj,
                                            self.features_nonezero, 
                                            dropout=0, activation='relu',
                                            input_shape=(None,self.input_dim))
        self.conv2 = ConvolutionLayer(self.hidden1,
                                        self.embedding_dim,
                                        self.adj,
                                        dropout=0, activation='relu')
        
        # GAE decoder
        self.reconstruct = InnerProductDecoder(self.embedding_dim,dropout=0,act=tf.nn.sigmoid)

    def call(self, inputs):
        '''
        A feedforward computation of embedding and reconstruct the Adjacent matrix from embeddings.
        '''
        
        indices= np.array(inputs[0])
        values = np.array(inputs[1])
        dense_shape = np.array(inputs[2])
        sparse_inputs = tf.SparseTensor(indices = indices,
                                        values = values,
                                        dense_shape = dense_shape)
        sparse_inputs = tf.sparse.reorder(sparse_inputs)
        inputs = sparse_inputs
        output1 = self.conv1(inputs)
        embedding = self.conv2(output1)
        a_hat = self.reconstruct(embedding)
        return embedding, a_hat

class WeightedConcateAggregator(keras.Model):
    '''
    The weighted concate aggregator concante embeddings from single views based on view weihts.
    Z = [ W_view1weight * Z1, W_view2weight * Z2, .... ]
    '''
    def __init__(self, input_dim, output_dim=1):
        super(WeightedConcateAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # The aggregator uses only one weighted layer to aggrate all embeddings.
        self.agg = WeightedConcateLayer(self.input_dim, activation='relu', output_dim = self.output_dim)

    def call(self, embs):        
        '''
        A feedforward computation of Aggregating all embeddings from single views to one embedding according to view weights
        '''
        embedding = self.agg(embs)
        return embedding

class Decoder(keras.Model):
    '''
    Simple Decoder using Z and Z^t
    initiates:
    input_dim = num of features, 
    hidden1_dim = num of first hidden layers neurons, 
    embedding_dim = embedding dimension, 
    features_nonezero.
    '''
    def __init__(self, embedding_dim):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        # Inner Product decoder
        self.reconstruct = InnerProductDecoder(self.embedding_dim, dropout=0, act=tf.nn.sigmoid)

    def call(self, inputs):
        '''
        Reconstruct the Adjacent matrix from embeddings.
        '''
        embedding = inputs
        a_hat = self.reconstruct(embedding)
        return a_hat

def loss_function(preds, targs, pos_weight, norm, sim_all, embedding, num_nodes, lam, loss_func):
    '''
    The loss function of MagCAE. Loss function takes:
    1. the original input as targets, 
    2. reconstructed adjacent matrix as predicts
    3. pairwise similarity
    4. network embedding
    5. penalty weights of each parts
    Loss_total = Sum(loss_nodes) + lam * pairwise similarity between nodes.
    '''
    loss = 0
    # reconstruction error
    reconstruct_loss = norm * tf.reduce_mean(
                        tf.nn.weighted_cross_entropy_with_logits(labels = targs, logits = preds,
                             pos_weight = pos_weight))
    if loss_func == 'all':
        embedding = tf.make_tensor_proto(embedding)
        embedding = tf.make_ndarray(embedding)

        pair_wise_loss = lam * cal_pairwise_loss(sim_all, embedding)
        loss = reconstruct_loss  + pair_wise_loss
    else:
        loss = reconstruct_loss
    return loss

def cal_attr_sim(views_feature_matrix, dataset, kernal="rbf"):

    product_sim = np.ones(shape=(views_feature_matrix['view1'].shape[0],views_feature_matrix['view1'].shape[0]), dtype = np.float32)

    if kernal == "rbf":
        for view in list(views_feature_matrix.keys()):
            # calculate rbf kernal sim in this view
            attr_sim = rbf_kernel(views_feature_matrix[view], views_feature_matrix[view])
            # use hadmard multiply
            product_sim = product_sim * attr_sim
    return product_sim

def cal_emb_sim(embedding, m="l2"):

    emb_sim = rbf_kernel(embedding,embedding)
    return emb_sim

def cal_pairwise_loss(sim_all, embedding, m="l2"):

    emb_sim = cal_emb_sim(embedding, m)
    distance = abs(sim_all.sum() - emb_sim.sum()) / (emb_sim.shape[0]*emb_sim.shape[0])
    pairwise_loss = distance
 
    return pairwise_loss

def pair_wise_view_sim(feature_matrix):
    
    dist = rbf_kernel(feature_matrix, feature_matrix)
    return dist

# This function is out of date, delete if published.
def pair_wise_sim(views_feature_matrix, dataset):
    """
    Take feature matrix from different views and calculate pairwise similarity as a inner product of similarity
    from each view.
    """
    views_sim = {}
    for view in list(views_feature_matrix.keys()):
        dist = pair_wise_view_sim(views_feature_matrix[view])
        views_sim.update({view:dist})
    
    product_sim = np.ones(shape=views_sim['view1'].shape, dtype = np.float32)
    for view in list(views_feature_matrix.keys()):
        product_sim *= views_sim[view]
    for i in range(len(product_sim)):
        if product_sim[i] >= 0.3:
            product_sim[i] = -1.
        else:
            product_sim[i] = 1.

    return product_sim

def aggregate_embeddings(embeddings, method):

    if method == "weighted_concat":
        aggregator = WeightedConcateAggregator(len(embeddings),1)
        embedding = aggregator(embeddings)
    
    if method == "others":
        pass

    return embedding, aggregator
