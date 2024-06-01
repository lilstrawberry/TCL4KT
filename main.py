import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math
import torch.optim as optim
import argparse
from load_data import *
from model import *
from run import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to test KT')
    parser.add_argument('--datapath', type=str, default='assistment2009',
                            help='dataset path')
    parser.add_argument('--T', type=int, default=2, help='select metapath_num')
    parser.add_argument('--batch_size', type=int,
                        default=80, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--embed', type=int, default=128,
                        help='the embedding size')
    parser.add_argument('--dropout', type=float,
                        default=0.3, help='the dropout ratio')
    params = parser.parse_args()












    datapath = params.datapath
    T = params.T

    meta_pro_pretrained, all_pro_num, all_skill_num, all_user_num, pro_skill_neibor, pro_user_neibor_true, pro_user_neibor_false, skill_feature, user_feature_true, user_feature_false, positive_sparse_tensor_matrix, other_feature_adj = load_need(datapath, T)

    # 没有邻居的随机选一个
    for i, x in enumerate(pro_user_neibor_true):
        if len(x) == 0:
            pro_user_neibor_true[i] += list(np.random.choice(np.array(all_user_num), 1))

    for i, x in enumerate(pro_user_neibor_false):
        if len(x) == 0:
            pro_user_neibor_false[i] += list(np.random.choice(np.array(all_user_num), 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')

    pretrained_meta_pro = meta_pro_pretrained
    pretrained_meta_pro = [torch.FloatTensor(x).to(device) for x in pretrained_meta_pro]

    dkt_lamda = 0.001
    pro_max = all_pro_num
    pro_begin_d = params.embed
    skill_max = all_skill_num
    user_max = all_user_num
    hidden = params.embed
    feature_drop = 0.0
    attn_drop = 0.0
    dropout = params.dropout

    other_neibor_list = [pro_skill_neibor, pro_user_neibor_true, pro_user_neibor_false]

    other_feature_list = [skill_feature, user_feature_true, user_feature_false]
    other_feature_list = [torch.FloatTensor(x).to(device) for x in other_feature_list]

    sample_list = [1, 5, 1]
    positive_matrix = positive_sparse_tensor_matrix
    positive_matrix = positive_matrix.to(device)
    other_feature_adj = [x.to(device) for x in other_feature_adj]
    tau = 0.8
    contrast_lambda = 0.1
    all_lamda = 0.5
    learning_rate = params.lr
    epochs = 40
    batch_size = params.batch_size
    min_seq = 3
    max_seq = 200
    grad_clip = 15.0
    patience = 20

    avg_auc = 0
    avg_acc = 0

    for now_step in range(5):

        best_acc = 0
        best_auc = 0
        state = {'auc': 0, 'acc': 0, 'loss': 0}

        model = TCL4KT(dkt_lamda, other_feature_adj, pretrained_meta_pro, pro_max, pro_begin_d, skill_max, user_max,
                             hidden, feature_drop, attn_drop, dropout, other_neibor_list, sample_list,
                             positive_matrix, tau, contrast_lambda)
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # , weight_decay=1e-5
        one_p = 0

        for epoch in range(epochs):

            train_path = f'data/{datapath}/{datapath}_train_question.txt'
            one_p += 1

            train_loss, train_acc, train_auc = run_epoch(all_lamda, other_feature_list, pro_max, train_path, batch_size,
                                                         True, min_seq, max_seq, model, optimizer, criterion, device,
                                                         grad_clip)
            print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

            test_path = f'data/{datapath}/{datapath}_test_question.txt'
            test_loss, test_acc, test_auc = run_epoch(all_lamda, other_feature_list, pro_max, test_path, batch_size, False,
                                                      min_seq, max_seq, model, optimizer, criterion, device, grad_clip)

            print(f'epoch: {epoch}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')

            if test_auc > best_auc:
                one_p = 0
                best_auc = test_auc
                best_acc = test_acc
                torch.save(model.state_dict(), f"./HGCLKT{now_step}_model.pkl")
                state['auc'] = test_auc
                state['acc'] = test_acc
                state['loss'] = test_loss
                torch.save(state, f'./HGCLKT{now_step}_state.ckpt')

            if one_p >= patience:
                break

        print(f'*******************************************************************************')
        print(f'best_acc: {best_acc:.4f}, best_auc: {best_auc:.4f}')
        print(f'*******************************************************************************')

        avg_auc += best_auc
        avg_acc += best_acc

    avg_auc = avg_auc / 5
    avg_acc = avg_acc / 5
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
