# -*- coding: utf-8 -*-
import numpy as np
import sys
import math
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset, num_views=3):
    '''
    load_data returns:
    1. adjacent matrix 
    2. features in different views, default is 3
    '''

    if dataset == 'cora':
        dataset_prefix = 'data/cora/cora'
    if dataset == 'citeseer':
        dataset_prefix = 'data/citeseer/citeseer'
    if dataset == 'Epinions':
        dataset_prefix = 'data/Epinions/Epinions'
    if dataset == 'Ciao':
        dataset_prefix = 'data/Ciao/Ciao'
    if dataset == 'ACM':
        dataset_prefix = 'data/ACM/ACM'

    if dataset == 'cora':
        objects = []
        formats = ['feat', 'graph']
        for i in range(len(formats)):
            # read those 4 types of input files
            with open(dataset_prefix +".{}".format(formats[i]),'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        
        all_features, graph = tuple(objects)
        views_features = {}    
        x , y = all_features.shape
        slice_size = math.ceil(y/num_views)
        start = 0
        for view in range(num_views):
            features = all_features[:,start:start+slice_size]
            start += slice_size
            views_features.update({"view"+ str(view+1):features})

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        return adj,views_features

    if dataset == 'citeseer':
        objects = []
        formats = ['feat', 'graph']
        for i in range(len(formats)):
            # read those 4 types of input files
            with open(dataset_prefix +".{}".format(formats[i]),'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        all_features, graph = tuple(objects)
        views_features = {}    
        x , y = all_features.shape
        slice_size = math.ceil(y/num_views)
        start = 0
        for view in range(num_views):
            features = all_features[:,start:start+slice_size]
            start += slice_size
            views_features.update({"view"+ str(view+1):features})
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        
        return adj,views_features

    if dataset == 'Epinions':
        
        formats = ['profile.feat','reviewed_items.feat','reviews.feat','graph']
        objects = []

        for i in range(len(formats)):
            # read those 4 types of input files
            with open(dataset_prefix +".{}".format(formats[i]),'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        feat_view1, feat_view2, feat_view3, graph = tuple(objects)

        views_features = {}
        views_features.update({"view1" : feat_view1})
        views_features.update({"view2" : feat_view2})
        views_features.update({"view3" : feat_view3})
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    if dataset == 'Ciao':
        
        records = ['review_items.feat','reviews.feat','user_profile.feat','graph']
        objects = []

        for r in records:
            # read those 4 types of input files
            with open(dataset_prefix +".{}".format(r),'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        feat_view1, feat_view2, feat_view3, graph = tuple(objects)
        views_features = {}
        views_features.update({"view1" : feat_view1})
        views_features.update({"view2" : feat_view2})
        views_features.update({"view3" : feat_view3})
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    if dataset == 'ACM':
        
        records = ['abstract.feat','title.feat','references.feat','graph']
        objects = []
        for r in records:
            # read those 4 types of input files
            with open(dataset_prefix +".{}".format(r),'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        feat_view1, feat_view2, feat_view3, graph = tuple(objects)
        views_features = {}
        views_features.update({"view1" : feat_view1})
        views_features.update({"view2" : feat_view2})
        views_features.update({"view3" : feat_view3})
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj,views_features

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape

def mask_test_edges(dataset, adj):

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # this matrix is symmetic, both adj_normalized and adj_normalized_ are the same.
    adj_normalized_ = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_inv_sqrt).tocoo()
    # A^hat = D^-1/2 * A * D ^-1/2
    return sparse_to_tuple(adj_normalized)


def lambda_test_edges(dataset, adj, l):
    # Function to build training test with proportion l 
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    test_ratio = 1 - 0.05 - l
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] * test_ratio))
    num_val = int(np.floor(edges.shape[0] * 0.05))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false