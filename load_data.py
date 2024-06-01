import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils import *
import torch.utils.data as data
import numpy as np

def getLoader(problem_max, path, batch_size, is_train, min_problem_num, max_problem_num):
    read_data = getReader(path)
    problem_list, ans_list = read_data.readData()
    dataset = HGCLKT_Dataset(problem_max, problem_list, ans_list, min_problem_num, max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader

class HGCLKT_Dataset(data.Dataset):
    def __init__(self, problem_max, problem_list, ans_list, min_problem_num, max_problem_num):
        self.problem_max = problem_max
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num
        self.problem_list, self.ans_list = [], []
        # 个人定义，少于 min_problem_num 丢弃
        # 根据论文 多于 max_problem_num  的分成多个 max_problem_num
        for (problem, ans) in zip(problem_list, ans_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]

                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    self.problem_list.append(item_problem)
                    self.ans_list.append(item_ans)
            else:
                item_problem = problem
                item_ans = ans
                self.problem_list.append(item_problem)
                self.ans_list.append(item_ans)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        now_problem = self.problem_list[index]
        now_problem = np.array(now_problem)

        now_ans = self.ans_list[index]
        # 由于需要统一格式
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[-num:] = now_problem
        use_ans[-num:] = now_ans

        next_ans = use_ans[1:]
        next_problem = use_problem[1:]
        last_ans = use_ans[:-1]
        last_problem = use_problem[:-1]

        use_mask[-num:] = 1
        next_mask = use_mask[1:]

        last_problem = torch.from_numpy(last_problem).to(device).long()
        next_problem = torch.from_numpy(next_problem).to(device).long()
        last_ans = torch.from_numpy(last_ans).to(device).long()
        next_ans = torch.from_numpy(next_ans).to(device).float()

        res_mask = torch.from_numpy(next_mask != 0).to(device)

        return last_problem, last_ans, next_problem, next_ans, res_mask

class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        ans_list = []
        split_char = ','

        read = open(self.path, 'r')
        for index, line in enumerate(read):
            if index % 3 == 0:
                pass

            elif index % 3 == 1:
                problems = line.strip().split(split_char)
                # 由于列表problems每个元素都是char 需要变为int
                problems = list(map(int, problems))
                problem_list.append(problems)

            elif index % 3 == 2:
                ans = line.strip().split(split_char)
                # 由于列表ans每个元素都是char 需要变为int
                ans = list(map(int, ans))
                ans_list.append(ans)

        read.close()
        return problem_list, ans_list

def getDf_user(ques_user):
    user_list = ques_user['stu'].values
    ques_list = ques_user['ques'].values
    ques_user_temp = list(set(tuple(zip(ques_list, user_list))))
    ques_user = np.zeros((2, len(ques_user_temp)), dtype=np.int64)
    for i, (x, y) in enumerate(ques_user_temp):
        ques_user[0][i] = x
        ques_user[1][i] = y
    return ques_user

def load_need(dataset, T):
    ques_skill = pd.read_csv(f'data/{dataset}/{dataset}_ques_skill.csv').values.T
    ques_user = pd.read_csv(f'data/{dataset}/{dataset}_stu_ques.csv')
    #     user_list = ques_user['stu'].values
    #     ques_list = ques_user['ques'].values
    #     ques_user_temp = list(set(tuple(zip(ques_list, user_list))))
    #     ques_user = np.zeros((2, len(ques_user_temp)), dtype=np.int64)
    #     for i, (x, y) in enumerate(ques_user_temp):
    #         ques_user[0][i] = x
    #         ques_user[1][i] = y
    ques_user_true = getDf_user(ques_user[ques_user['correct'] == 1])
    ques_user_wrong = getDf_user(ques_user[ques_user['correct'] == 0])

    # --------------------------------------------------------------------------------
    all_pro_num = max(ques_skill[0]) + 1
    all_skill_num = max(ques_skill[1]) + 1
    all_user_num = max(ques_user_true[1]) + 1
    # --------------------------------------------------------------------------------
    pro_skill_neibor = []

    pro_user_neibor_true = []
    pro_user_neibor_false = []

    for x in range(all_pro_num):
        pro_skill_neibor.append(set())
        pro_user_neibor_true.append(set())
        pro_user_neibor_false.append(set())

    for (x, y) in zip(ques_skill[0, :], ques_skill[1, :]):
        pro_skill_neibor[x].add(y)

    for (x, y) in zip(ques_user_true[0, :], ques_user_true[1, :]):
        pro_user_neibor_true[x].add(y)

    for (x, y) in zip(ques_user_wrong[0, :], ques_user_wrong[1, :]):
        pro_user_neibor_false[x].add(y)

    for x in range(all_pro_num):
        pro_skill_neibor[x] = list(pro_skill_neibor[x])
        pro_user_neibor_true[x] = list(pro_user_neibor_true[x])
        pro_user_neibor_false[x] = list(pro_user_neibor_false[x])
    # --------------------------------------------------------------------------------
    skill_feature = sp.eye(all_skill_num)
    user_feature = sp.eye(all_user_num)

    skill_feature = preprocess_features(skill_feature)
    user_feature = preprocess_features(user_feature)
    # --------------------------------------------------------------------------------
    pro_skill_metapath_pretrained = np.load(f'data/{dataset}/{dataset}_qkq_contDiff_10_80_128_5.emb.npy')
    pro_user_metapath_pretrained = np.load(f'data/{dataset}/{dataset}_quq_cw_10_80_128_5.emb.npy')
    meta_pro_pretrained = [pro_skill_metapath_pretrained, pro_user_metapath_pretrained]
    # --------------------------------------------------------------------------------
    metapath_pro_skill = pd.read_csv(f'data/{dataset}/{dataset}_walks_qkq_contDiff_10_80.txt', sep=',',
                                     header=None).values
    metapath_pro_user = pd.read_csv(f'data/{dataset}/{dataset}_walks_quq_cw_10_80.txt', sep=',',
                                    header=None).values
    # --------------------------------------------------------------------------------
    metapath_pro_skill_set = []
    metapath_pro_user_set = []

    for x in range(all_pro_num):
        metapath_pro_skill_set.append(set())
        metapath_pro_user_set.append(set())

    for item_metapath_pro_skill in metapath_pro_skill:
        for now_index in range(len(item_metapath_pro_skill) - 1):
            now_pro = item_metapath_pro_skill[now_index]
            neibor_pro = item_metapath_pro_skill[now_index + 1]
            metapath_pro_skill_set[now_pro].add(neibor_pro)
            metapath_pro_skill_set[neibor_pro].add(now_pro)

    for item_metapath_pro_user in metapath_pro_user:
        for now_index in range(len(item_metapath_pro_user) - 1):
            now_pro = item_metapath_pro_user[now_index]
            neibor_pro = item_metapath_pro_user[now_index + 1]
            metapath_pro_user_set[now_pro].add(neibor_pro)
            metapath_pro_user_set[neibor_pro].add(now_pro)

    pro_skill_adj = create_metapath_adj(metapath_pro_skill_set)
    pro_user_adj = create_metapath_adj(metapath_pro_user_set)
    other_feature_adj = [pro_skill_adj, pro_user_adj]
    # --------------------------------------------------------------------------------
    pro_neibor_metapath_num = []
    metapath_neibor_num_yuzhi = T

    for x in range(all_pro_num):

        pro_neibor_metapath_num.append([])

        now_dict = dict()

        for skillpath_neibor in metapath_pro_skill_set[x]:
            if skillpath_neibor == x:
                continue

            if skillpath_neibor in now_dict:
                now_dict[skillpath_neibor] += 1
            else:
                now_dict[skillpath_neibor] = 1
        for userpath_neibor in metapath_pro_user_set[x]:
            if userpath_neibor in now_dict:
                now_dict[userpath_neibor] += 1
            else:
                now_dict[userpath_neibor] = 1
        for keys in now_dict.keys():
            if now_dict[keys] >= metapath_neibor_num_yuzhi:
                pro_neibor_metapath_num[x].append(keys)
    # --------------------------------------------------------------------------------
    positive_row = []
    positive_col = []
    for x in range(len(pro_neibor_metapath_num)):
        now_neibor = pro_neibor_metapath_num[x]
        positive_row += [x for _ in range(len(now_neibor))]
        positive_col += now_neibor

    positive_indices = torch.from_numpy(np.vstack([np.array(positive_row), np.array(positive_col)]).astype(np.int64))
    positive_values = torch.ones(len(positive_row)).long()
    positive_size = torch.Size((all_pro_num, all_pro_num))
    positive_sparse_tensor_matrix = torch.sparse.FloatTensor(positive_indices, positive_values, positive_size)

    return meta_pro_pretrained, all_pro_num, all_skill_num, all_user_num, pro_skill_neibor, pro_user_neibor_true, pro_user_neibor_false, skill_feature, user_feature, user_feature, positive_sparse_tensor_matrix, other_feature_adj
