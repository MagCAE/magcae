# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import sys
import time
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle as pkl
import pandas as pd
from scipy.sparse import csc_matrix
from Model import GAE, Decoder, loss_function, aggregate_embeddings, pair_wise_sim, pair_wise_view_sim, cal_attr_sim
from preprocessing import load_data, mask_test_edges, preprocess_graph, sparse_to_tuple, lambda_test_edges
from evaluation import evaluate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

def parse_args():
    '''
    Parses the MagCAE arguments.
    '''
    parser = argparse.ArgumentParser("MagCAE",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='the name of dataset'
    )	
    parser.add_argument('--output', nargs='?', default='output/emb',
                        help='Saved embedding file path')

    parser.add_argument('--agg', nargs='?', default='weighted_concat',
                        help='Aggregation Function')

    parser.add_argument('--epochs', default=65, type=int,
                        help='Number of epochs to Train')
    
    parser.add_argument('--p', default=0.3, type=float,
                        help='Proportion of embedding dimension / input dimension')

    parser.add_argument('--tr', default=0.85, type=float,
                        help='Training ratio of edges')

    parser.add_argument('--lam', default=1, type=float,
                        help='Pairwise loss coefficient lambda')

    parser.add_argument('--loss', nargs='?', default='all',
                        help='the loss function, \'all\' for the reconstruction loss + pairwise similarity loss; \
                        \'res\' for reconstruction loss only')

    return parser.parse_args()

def main(args):
    
    dataset = args.dataset
    emb_output_dir = args.output
    epochs = args.epochs
    agg = args.agg
    p = args.p
    tr = args.tr
    lam = args.lam
    lose_func = args.loss

    # Preprocess dataset
    adj, views_features = load_data(dataset, num_views=3)
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # Calculate pairwise simlarity.
    views_sim_matrix = {}
    views_feature_matrix = {}

    for view in list(views_features.keys()):
        feature_matrix = csc_matrix.todense(views_features[view])
        views_feature_matrix.update({view:feature_matrix})
 
    kernal = "rbf"
    if lose_func == 'all':
        attr_sim = cal_attr_sim(views_feature_matrix, dataset)
    else:
        attr_sim = 0

    # split nodes to train, valid and test datasets, 
    # remove test edges from train adjacent matrix. 
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(dataset, adj)
    
    print("Masking edges Done!")
    adj = adj_train
    nx_G = nx.from_numpy_array(adj.toarray())
    num_nodes = adj.shape[0]
    adj_norm = preprocess_graph(adj)

    views_features_num = {}
    views_features_nonzero = {}
    for view in list(views_features.keys()):
        views_features[view] = sparse_to_tuple(views_features[view].tocoo())
        views_features_num.update({view:views_features[view][2][1]})
        views_features_nonzero.update({view:views_features[view][1].shape[0]})
    
    # Build model
    MagCAE = {}
    for view in list(views_features.keys()):
        x,y = views_features[view][2][0], views_features[view][2][1]
        model = GAE(y, views_features_nonzero[view], adj_norm, math.ceil(2*p*y), math.ceil(p*y))
        MagCAE.update({view:model})

    # Loss function and optimizer.
    # loss weight taken by each nodes to the total loss.
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) /adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float(adj.shape[0] * adj.shape[0] - adj.sum())*2
    optimizer = tf.keras.optimizers.Adam()

    adj_targ = adj_train + sp.eye(adj_train.shape[0])
    adj_targ = sparse_to_tuple(adj_targ)

    indices= np.array(adj_targ[0])
    values = np.array(adj_targ[1])
    dense_shape = np.array(adj_targ[2])
    sparse_targ = tf.SparseTensor(indices = indices,
                                    values = values,
                                    dense_shape = dense_shape)
    sparse_targ = tf.cast(sparse_targ, dtype=tf.float32)

    adj_targ = tf.sparse.to_dense(sparse_targ)
    adj_targ = tf.reshape(adj_targ,[-1])
    # Train and Evaluate Model
    # Training Loop:
    # In each epoch: views - > view_embedding -> aggregate embedding -> total loss ->  update gradients
    decoder = Decoder(100)

    for epoch in range(epochs):
        loss = 0
        start = time.time()

        with tf.GradientTape() as tape:
            ag_embedding ={}


            for VAE in list(MagCAE.keys()):
                v_embedding, a_hat = MagCAE[VAE](views_features[VAE])
                ag_embedding.update({VAE:v_embedding})

            # aggregate embeddings
            embedding, aggregator = aggregate_embeddings(ag_embedding, agg)
            # reconstruct a_hat
            a_hat = decoder(embedding)
            loss += loss_function(a_hat, adj_targ, pos_weight, norm, attr_sim, embedding, num_nodes, lam, lose_func)

        if agg == "weighted_concat":
            variables = MagCAE['view1'].trainable_variables + MagCAE['view2'].trainable_variables + MagCAE['view3'].trainable_variables + aggregator.trainable_variables

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        # Evaluate on validate set
        embedding = np.array(embedding)
        roc_cur, ap_cur, _, _ = evaluate(val_edges, val_edges_false, adj_orig, embedding)

        print("Epoch {}: Val_Roc {:.4f}, Val_AP {:.4f}, Time Consumed {:.2f} sec\n".format(epoch+1, roc_cur, ap_cur, time.time()-start))

    print("Training Finished!")
    
    # Evaluation Result on test Edges
    test_embedding= {}
    for VAE in list(MagCAE.keys()):
        v_embedding, a_hat = MagCAE[VAE](views_features[VAE])
        test_embedding.update({VAE:v_embedding})

    # aggregate embeddings
    embedding, aggregator = aggregate_embeddings(test_embedding, agg)
    embedding = np.array(embedding) # embedding is a tensor, convert to np array.

    # reconstruct a_hat
    test_roc, test_ap, fpr, tpr = evaluate(test_edges, test_edges_false, adj_orig, embedding)
    print("MagCAE test result on {}".format(dataset))
    print("Test Roc: {}, Test AP: {}, P: {}, Training Ratio: {}, Lambda: {}.".format(test_roc, test_ap, p, tr, lam))

if __name__ == "__main__":
    args = parse_args()
    main(args)