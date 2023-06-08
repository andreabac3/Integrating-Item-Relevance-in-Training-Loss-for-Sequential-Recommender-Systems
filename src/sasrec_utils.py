import sys
import os
import copy
#import torch
import random
import numpy as np
#from collections import defaultdict
from multiprocessing import Process, Queue
import pickle as pkl
import json

from collections import defaultdict
 
from scipy.stats import kendalltau

from utils import compute_single_rls

from utils import write_json

from dataclasses import dataclass

@dataclass
class SASRecMetrics:
    ndcg_5: float
    ndcg_10: float
    ndcg_15: float
    ndcg_20: float

    ndcg_5_rel: float
    ndcg_10_rel: float
    ndcg_15_rel: float
    ndcg_20_rel: float

    hr_5: float
    hr_10: float
    hr_15: float
    hr_20: float

    rls_rbo: float
    rls_jac: float


#from data_creation import get_rating_files_per_dataset

def random_neq_FS(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function_FS(user_train, train_user_keys, itemnum, batch_size, maxlen, num_positives, num_negatives, result_queue, SEED):
    def sample():
        user = np.random.choice(train_user_keys)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen,num_positives], dtype=np.int32)
        neg = np.zeros([maxlen,num_negatives], dtype=np.int32)

        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])  # ts = total sequence. it is a set of all the items that a user had a interaction
        nxts = [nxt]  # 
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            for j in range(min(num_positives, len(nxts))):  # in this way, positives are added in order: from nearest step, to furthest step
                pos[idx,j] = nxts[-j-1]
            if pos[idx,0] !=0:
                for j in range(num_negatives):
                    neg[idx,j] = random_neq_FS(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            nxts.append(nxt)
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = [sample() for i in range(batch_size)]

        result_queue.put(zip(*one_batch))


class WarpSampler_FS(object):
    def __init__(self, user_train, itemnum, batch_size=64, maxlen=10, n_workers=1, num_positives=1, num_negatives=1): ###FEDSIC ADDITION: if sampling negative items also from seq
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        self.train_user_keys = list(user_train.keys())

        # poss_neg_items_per_user = {}
        # for user in self.train_user_keys:
        #     #if sample_from_seq: ###FEDSIC ADDITION: if sampling negative items also from seq #sample_from_seq=False, 
        #     poss_neg_items_per_user[user] = list(set(range(1, itemnum + 1)).difference(user_train[user]))

        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function_FS, args=(user_train,
                                                      self.train_user_keys,
                                                      itemnum,
                                                      #poss_neg_items_per_user,
                                                      batch_size,
                                                      maxlen,
                                                      num_positives,
                                                      num_negatives,
                                                      self.result_queue,
                                                      201094
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()







# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
# def data_partition(fname):
#     usernum = 0
#     itemnum = 0
#     User = defaultdict(list)
#     user_train = {}
#     user_valid = {}
#     user_test = {}
#     # assume user/item index starting from 1
#     f = open('../data/raw/'+fname+"/"+get_rating_files_per_dataset(fname)[0], 'r')
#     for line in f:
#         u, i = line.rstrip().split(' ')
#         u = int(u)
#         i = int(i)
#         usernum = max(u, usernum)
#         itemnum = max(i, itemnum)
#         User[u].append(i)

#     for user in User:
#         nfeedback = len(User[user])
#         if nfeedback < 3:
#             user_train[user] = User[user]
#             user_valid[user] = []
#             user_test[user] = []
#         else:
#             user_train[user] = User[user][:-2]
#             user_valid[user] = []
#             user_valid[user].append(User[user][-2])
#             user_test[user] = []
#             user_test[user].append(User[user][-1])
#     return [user_train, user_valid, user_test, usernum, itemnum]

def load_pickle_dataset(project_folder,experiment_id):
    dataset_loc = os.path.join(project_folder,"data","processed",f"{str(experiment_id)}.pkl")
    with open(dataset_loc,'rb') as f:
        dataset = pkl.load(f)
    return dataset

def compute_eval_metric(metric_name, ranks, metric_at, item_relevance=False):
    if metric_name == "ndcg":
        return compute_ndcg(ranks, metric_at, item_relevance)
    elif metric_name == "hr":
        return compute_hr(ranks, metric_at)
    else:
        raise NotImplementedError

def compute_ndcg(ranks, metric_at, use_item_relevance=False):
    app_dcg, app_hr = 0,0
    n_ranks = len(ranks)
    for idx, rank in enumerate(ranks):
        item_relevance = n_ranks - idx
        if rank < metric_at: #at 10
            if use_item_relevance:
                app_dcg += item_relevance / np.log2(rank + 2)
            else:
                app_dcg += 1 / np.log2(rank + 2)
            app_hr += 1
        else: break
    app_idcg = np.sum([1 / np.log2(r + 2) if not use_item_relevance else (len(ranks)-r)/ np.log2(r + 2) for r in range(min(len(ranks),metric_at))]) #/IDCG
    ndcg = app_dcg / app_idcg
    return ndcg

def compute_hr(ranks, metric_at) -> float:
    return sum([1 for rank in ranks if rank < metric_at]) / len(ranks)

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, train, valid, test, usernum, itemnum, args, use_test = True, metrics_loc=None):
    metric_names = ["ndcg","hr"]
    cutoffs = [5,10,15,20]
    item_relevances = [True,False]

    metrics = defaultdict(list)
    #metrics = {"_".join([metric_name,str(cutoff)]):[] for metric_name in metric_names for cutoff in cutoffs}
    # metrics["rls_rbo"] = []
    # metrics["rls_jacc"] = []
    # metrics["ktau_corr"] = []
    # metrics["ktau_pval"] = []
    
    valid_user = 0

    if use_test:
        data_to_use = test
    else:
        data_to_use = valid

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(data_to_use[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if use_test:
            seq[idx] = valid[u][0]
            idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = data_to_use[u].copy()
        for _ in range(101-len(data_to_use[u])):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        ranked_list_indices = predictions.argsort().detach().cpu()
        ranks = ranked_list_indices.argsort()[:len(data_to_use[u])].detach().cpu().numpy()#.item()
        
        valid_user += 1

        # dcg = sum([1 / np.log2(rank + 2) for rank in ranks if rank < 10])
        # idcg = sum([1 / np.log2(rank + 2) for rank in range(10)])
        # hit_rate = sum([1 for rank in ranks if rank < 10])
        # NDCG += dcg / idcg
        # HT += hit_rate / len(ranks)
        
        # app_dcg, app_hr = 0,0
        # for rank in ranks:
        #     if rank < 10: #at 10
        #         app_dcg += 1 / np.log2(rank + 2)
        #         app_hr += 1
        # app_idcg = np.sum([1 / np.log2(r + 2) for r in range(min(len(ranks),10))]) #/IDCG
        #NDCG += app_dcg / app_idcg
        #HT += app_hr / len(ranks)

        for metric_name in metric_names:
            for cutoff in cutoffs:
                if metric_name == "ndcg":
                    for item_relevance in item_relevances:
                        save_name = "_".join([metric_name,str(cutoff)])
                        if item_relevance: save_name = save_name+"_rel"
                        metrics[save_name].append(compute_eval_metric(metric_name, ranks, cutoff, item_relevance))
                else:
                    metrics["_".join([metric_name,str(cutoff)])].append(compute_eval_metric(metric_name, ranks, cutoff))
        ranked_list = [item_idx[i] for i in ranked_list_indices]
        app_rbo, app_jac = compute_single_rls(ranked_list[:len(data_to_use[u])],item_idx[:len(data_to_use[u])], metrics_at=[10])
        app_rbo = app_rbo[0]; app_jac = app_jac[0]
        metrics["rls_rbo"].append(app_rbo)
        metrics["rls_jac"].append(app_jac)

        #compute kendall's tau
        # app_ktau = kendalltau(ranked_list[:len(data_to_use[u])],item_idx[:len(data_to_use[u])])
        # app_ktau_corr = app_ktau.correlation
        # app_ktau_pvalue = app_ktau.pvalue

        # metrics["ktau_corr"].append(app_ktau_corr)
        # metrics["ktau_pval"].append(app_ktau_pvalue)

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    #dump all metrics to file
    if metrics_loc is not None:
        write_json(metrics_loc,metrics)

    for metric_name in metrics:
        metrics[metric_name] = sum(metrics[metric_name])/len(metrics[metric_name])

    return SASRecMetrics(**metrics)
    #return NDCG / valid_user, HT / valid_user, RLS_RBO / valid_user, RLS_JAC / valid_user