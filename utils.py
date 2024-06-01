import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math

def preprocess_features(features):
    # 正规化特征 features: n feature
    row_sum = np.array(features.sum(1))
    row_inv = np.power(row_sum, -1).flatten()
    # n
    row_inv[np.isinf(row_inv)] = 0
    sparse_diag = sp.diags(row_inv)
    # n n
    res = sparse_diag.dot(features).todense()
    # n feature
    return res

def normalize_adj(adj):
    # 正规化稀疏矩阵
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    sparse_diag = sp.diags(d_inv_sqrt)
    res = adj.dot(sparse_diag).transpose().dot(sparse_diag).tocoo()
    # D^(-0.5) A D^(0.5)
    return res

def sparse_mx_to_torch_sparse_tensor(sparse_matrix):
    # 稀疏矩阵转为torch的稀疏矩阵
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack([sparse_matrix.row, sparse_matrix.col]).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data)
    size = torch.Size(sparse_matrix.shape)
    res = torch.sparse.FloatTensor(indices, values, size)
    return res

def create_metapath_adj(metapath_list):
    indices_x = []
    indices_y = []
    degree_pro = []
    for item in metapath_list:
        degree_pro.append(len(item))
    values = []
    for i, item in enumerate(metapath_list):
        indices_x.append(i)
        indices_y.append(i)
        values.append(1.0 / (1 + degree_pro[i]))
        for item_one in item:
            indices_x.append(i)
            indices_y.append(item_one)
            values.append(1.0 / (math.sqrt(1 + degree_pro[i]) * math.sqrt(1 + degree_pro[item_one])))
    indices = torch.from_numpy(
        np.vstack([np.array(indices_x), np.array(indices_y)]).astype(np.int64)
    )
    values = torch.from_numpy(np.array(values)).float()
    size = torch.Size((len(metapath_list), len(metapath_list)))
    res = torch.sparse.FloatTensor(indices, values, size)
    return res