import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import math

class Sc_Attention(nn.Module):
    def __init__(self, hidden, attn_drop):
        super(Sc_Attention, self).__init__()

        self.attn = nn.Parameter(torch.rand(2 * hidden, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, pro_feature, neibor_feature, now_neibor_list):
        # pro_feature: n hidden
        # neibor_feature: type_num hidden
        # now_neibor_list: n type_sample_neibor_num
        now_neibor_feature = F.embedding(now_neibor_list, neibor_feature)
        # n type_sample_neibor_num hidden
        now_feature = pro_feature.unsqueeze(1).expand_as(now_neibor_feature)
        # n type_sample_neibor_num hidden

        all_feature = torch.cat([now_feature, now_neibor_feature], dim=-1)
        # n type_sample_neibor_num 2*hidden

        attn = self.attn_drop(self.attn)
        attn_mark = torch.matmul(all_feature, attn)
        # n type_sample_neibor_num 1
        attn_mark = F.leaky_relu(attn_mark)
        attn_mark = torch.softmax(attn_mark, dim=1)
        # n type_sample_neibor_num 1
        now_type_feature = (now_neibor_feature * attn_mark).sum(1)
        # n hidden
        return now_type_feature

class Sc_encoder(nn.Module):
    def __init__(self, hidden, attn_drop, other_neibor_list, sample_list):
        super(Sc_encoder, self).__init__()

        self.other_neibor_list = other_neibor_list
        self.sample_list = sample_list

        self.Sc_Attention_list = nn.ModuleList([
            Sc_Attention(hidden, attn_drop)
            for _ in range(len(sample_list))
        ])

        self.sc_linear = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.sc_linear.weight, gain=1.414)

        self.beta_attn = nn.Parameter(torch.rand(hidden, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.beta_attn.data, gain=1.414)

        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, pro_feature, other_features_list):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')

        all_type_feature = []

        for i in range(len(self.sample_list)):

            now_need_sample_num = self.sample_list[i]
            now_neibor_list = []
            for now_neibor in self.other_neibor_list[i]:
                if len(now_neibor) > now_need_sample_num:
                    sample_neibor = torch.from_numpy(np.random.choice(now_neibor, now_need_sample_num, replace=False))
                else:
                    sample_neibor = torch.from_numpy(np.random.choice(now_neibor, now_need_sample_num, replace=True))
                sample_neibor = sample_neibor.unsqueeze(0).to(device)
                # 1 neibor
                now_neibor_list.append(sample_neibor)

            now_neibor_list = torch.cat(now_neibor_list, dim=0)
            # n sample_neibor_num
            item_type_feature = self.Sc_Attention_list[i](pro_feature, other_features_list[i], now_neibor_list)
            # n hidden
            all_type_feature.append(item_type_feature)

        now_beta_attn = self.attn_drop(self.beta_attn)
        # hidden 1
        beta_attn = []
        for now_item_feature in all_type_feature:
            # now_item_feature: n hidden
            now_type_feature = torch.tanh(self.sc_linear(now_item_feature)).mean(0)
            # hidden
            now_attn = torch.matmul(now_type_feature, now_beta_attn)
            # 1
            beta_attn.append(now_attn)
        beta_attn = torch.cat(beta_attn, dim=0).view(-1)
        beta_attn = torch.softmax(beta_attn, dim=-1)

        z_feature = 0
        for i in range(len(all_type_feature)):
            z_feature = z_feature + beta_attn[i] * all_type_feature[i]
        return z_feature

class GCN(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GCN, self).__init__()

        self.fc = nn.Linear(in_feature, out_feature, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

        self.activation = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.rand(out_feature), requires_grad=True)
            self.bias.data.fill_(0.0)
        else:
            self.bias = None

    def forward(self, x, adj):
        # x: n in_feature
        # adj: n n
        x = self.fc(x)
        # n out_feature
        out = torch.spmm(adj, x)
        # n out_feature
        if self.bias is not None:
            out = out + self.bias
        out = self.activation(out)
        return out

class Mp_encoder(nn.Module):
    def __init__(self, hidden, attn_drop):
        super(Mp_encoder, self).__init__()

        self.fc_linear = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.fc_linear.weight)

        self.gcns = nn.ModuleList(
            [
                GCN(hidden, hidden)
                for _ in range(3)
            ]
        )

        self.beta_attn = nn.Parameter(torch.rand(hidden, 1), requires_grad=True)
        nn.init.xavier_uniform_(self.beta_attn.data, gain=1.414)

        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, pro_feature, other_feature_adj):
        # meta_pro: type_num n hidden
        meta_pro = []

        for index, item_adj in enumerate(other_feature_adj):
            item_embeds = self.gcns[index](pro_feature, item_adj)
            meta_pro.append(item_embeds)

        all_attn = []
        beta_attn = self.attn_drop(self.beta_attn)
        # hidden 1
        for now_meta in meta_pro:
            item_meta_embed = torch.tanh(self.fc_linear(now_meta)).mean(0)
            # hidden
            now_attn = torch.matmul(item_meta_embed, beta_attn)
            # 1
            all_attn.append(now_attn)

        all_attn = torch.cat(all_attn, dim=-1).view(-1)
        all_attn = torch.softmax(all_attn, dim=-1)

        z_mp = 0
        for i in range(len(meta_pro)):
            z_mp = z_mp + all_attn[i] * meta_pro[i]
        return z_mp

class ContrastLoss(nn.Module):
    def __init__(self, hidden, tau, contrast_lamda, positive_matrix):
        super(ContrastLoss, self).__init__()

        self.tau = tau
        self.contrast_lamda = contrast_lamda
        self.positive_matrix = positive_matrix

        self.project_linear = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden)
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        res = torch.matmul(z1, z2.transpose(-1, -2))
        res = torch.exp(res / self.tau)
        return res

    def forward(self, x1, x2):
        # x1: n hidden
        # x2: n hidden
        now_device = x1.device
        last_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #last_device = torch.device('cpu')

        if now_device != last_device:
            z1 = self.project_linear(x1.to(last_device)).to(now_device)
            z2 = self.project_linear(x2.to(last_device)).to(now_device)
        else:
            z1 = self.project_linear(x1)
            z2 = self.project_linear(x2)

        sc_mp = self.sim(z1, z2)
        mp_sc = self.sim(z2, z1)

        postive_matrix = self.positive_matrix.to(x1.device)

        sc_mp = sc_mp / (torch.sum(sc_mp, dim=-1, keepdim=True) + 1e-8)
        sc_loss = -torch.log((sc_mp * postive_matrix.to_dense()).sum(-1) + 1e-8).mean()

        mp_sc = mp_sc / (torch.sum(mp_sc, dim=-1, keepdim=True) + 1e-8)
        mp_loss = -torch.log((mp_sc * postive_matrix.to_dense()).sum(-1) + 1e-8).mean()

        contrast_loss = sc_loss * self.contrast_lamda + mp_loss * (1 - self.contrast_lamda)

        return contrast_loss.to(last_device)

class TCL4KT(nn.Module):
    def __init__(self, dkt_contrast_lamda, other_feature_adj, pretrained_meta_pro, pro_max, pro_begin_d, skill_max,
                 user_max,
                 hidden, feature_drop, attn_drop, dropout, other_neibor_list, sample_list,
                 positive_matrix, tau, contrast_lambda
                 ):

        super(TCL4KT, self).__init__()

        self.dkt_contrast_lamda = dkt_contrast_lamda
        self.other_feature_adj = other_feature_adj
        # self.pretrained_meta_pro = nn.Parameter(pretrained_meta_pro, requires_grad=True)
        self.pretrained_meta_pro = pretrained_meta_pro

        self.other_neibor_list = other_neibor_list
        self.sample_list = sample_list

        self.pro_feature = nn.Parameter(torch.rand(pro_max, pro_begin_d), requires_grad=True)

        self.projection = nn.ModuleList([
            nn.Linear(pro_begin_d, hidden),
            nn.Linear(skill_max, hidden),
            nn.Linear(user_max, hidden),
            nn.Linear(user_max, hidden)
        ])
        self.feature_drop = nn.Dropout(p=feature_drop)

        self.Sc_encoder = Sc_encoder(hidden, attn_drop, other_neibor_list, sample_list)
        self.Mp_encoder = Mp_encoder(hidden, attn_drop)
        self.ContrastLoss = ContrastLoss(hidden, tau, contrast_lambda, positive_matrix)
        #self.DKT_contrastLoss1 = DKTContrastLoss(hidden, tau)
        #self.DKT_contrastLoss2 = DKTContrastLoss(hidden, tau)
        # 这里用共享，后面不用共享试试

        self.dkt_c_embed = nn.Parameter(torch.rand(pro_max, hidden), requires_grad=True)
        self.dkt_u_embed = nn.Parameter(torch.rand(pro_max, 1), requires_grad=True)
        self.dkt_d_embed = nn.Parameter(torch.rand(pro_max, hidden), requires_grad=True)

        self.mp_linear = nn.Linear(2 * hidden, 1 * hidden)

        self.f1_linear = nn.Linear(hidden, 1)
        self.f2_linear = nn.Linear(2 * hidden, 1)

        self.lstm = nn.LSTM(2 * hidden, 2 * hidden, batch_first=True)
        self.final_predict = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden, 1)
        )

        self.dropout = nn.Dropout(p=dropout)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.pro_feature)
        nn.init.xavier_uniform_(self.dkt_c_embed)
        nn.init.xavier_uniform_(self.dkt_u_embed)
        nn.init.xavier_uniform_(self.dkt_d_embed)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, last_pro, last_ans, next_pro, other_feature_list):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')

        pro_feature = self.pro_feature
        # pro_feature = preprocess_features(pro_feature.cpu().detach().numpy())
        # pro_feature = self.pretrained_meta_pro

        # pro_feature = F.elu(self.mp_linear(torch.cat([self.pretrained_meta_pro[0], self.pretrained_meta_pro[1]], dim=-1)))

        pro_feature = F.elu(self.feature_drop(self.projection[0](pro_feature)))

        other_features_list = []

        for i in range(len(other_feature_list)):
            item_feature = F.elu(self.feature_drop(self.projection[i + 1](other_feature_list[i])))
            other_features_list.append(item_feature)

        # pretrained_meta_pro = self.pretrained_meta_pro

        sc_out = self.Sc_encoder(pro_feature, other_features_list)
        # pro_max hidden

        other_feature_adj = self.other_feature_adj
        mp_out = self.Mp_encoder(pro_feature, other_feature_adj)
        # mp_out = self.Mp_encoder(pretrained_meta_pro)
        # pro_max hidden
        dkt_embed = self.dkt_c_embed + self.dkt_u_embed * self.dkt_d_embed

        # f1_dkt_loss = self.DKT_contrastLoss1(dkt_embed, sc_out)
        f1_dkt_loss = 0
        # f2_dkt_loss = self.DKT_contrastLoss2(dkt_embed, mp_out)
        # self.dkt_contrast_lamda * (f1_dkt_loss + f2_dkt_loss)
        contrast_loss = self.ContrastLoss(sc_out, mp_out)

        #DEVICE = 'cuda:1'
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #DEVICE = torch.device('cpu')

        dkt_embed = dkt_embed.to(DEVICE)
        mp_out = mp_out.to(DEVICE)
        fffff_loss = self.dkt_contrast_lamda * self.ContrastLoss(dkt_embed, mp_out)
        contrast_loss = contrast_loss + fffff_loss

        mp_out = mp_out.to(device)
        # mp_out = dkt_embed
        # contrast_loss = contrast_loss + self.ContrastLoss(sc_out, pre_trained_out)
        # contrast_loss = 0
        # mp_out = torch.cat([mp_out, pretrained_meta_pro[0], pretrained_meta_pro[1]], dim=-1)

        # k1 = torch.tanh(self.f1_linear(mp_out))
        # k2 = torch.tanh(self.f2_linear(pre_trained_out))
        # mp_out = k1 * mp_out + pre_trained_out

        # mp_out = torch.cat([sc_out, mp_out], dim=-1)
        #         pre_trained_out = torch.cat([pretrained_meta_pro[0], pretrained_meta_pro[1]], dim=-1)
        #         pre_trained_out = self.mp_linear(pre_trained_out)
        #         k1 = torch.sigmoid(self.f1_linear(F.relu(pre_trained_out)))

        #         mp_out = (1-k1) * torch.tanh(mp_out) + k1 * pre_trained_out

        # mp_out = torch.cat([sc_out, mp_out], dim=-1)

        last_pro_embed = F.embedding(last_pro, mp_out)
        #         fff = F.embedding(last_pro, pre_trained_out)
        #         k1 = torch.sigmoid(self.f1_linear(fff))
        #         last_pro_embed = (1-k1) * torch.tanh(last_pro_embed) + k1 * torch.tanh(fff)

        # batch seq hidden
        next_pro_embed = F.embedding(next_pro, mp_out)
        # batch seq hidden
        Q = torch.cat([last_pro_embed, last_pro_embed], dim=-1)
        # batch seq 2*hidden
        last_ans_one_hot = F.one_hot(last_ans, 2).float()
        # batch seq 2
        need_matrix = torch.from_numpy(np.zeros((2, Q.shape[-1]))).to(device)
        need_matrix[0, last_pro_embed.shape[-1]:] = 1.0
        need_matrix[1, :last_pro_embed.shape[-1]] = 1.0
        last_ans_embed = torch.matmul(last_ans_one_hot, need_matrix.float())
        # batch seq 2*hidden
        now_Q = Q * last_ans_embed
        # batch seq 2*hidden
        now_Q = self.dropout(now_Q)
        # now_Q = self.transformer_encoder(now_Q)

        ls_out, _ = self.lstm(now_Q)
        # batch seq 2*hidden
        final_in = torch.cat([ls_out, next_pro_embed], dim=-1)
        # batch seq 6*hidden
        final_P = self.final_predict(final_in)
        # batch seq 1
        final_P = torch.sigmoid(final_P).squeeze(-1)
        # batch seq
        return final_P, contrast_loss