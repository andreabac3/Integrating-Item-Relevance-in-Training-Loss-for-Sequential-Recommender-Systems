

from typing import *
import re
import numpy as np
from Levenshtein import distance as lev
from utils import *



def parse_gpt_output(txt: str) -> List[str]:
    #app = [x.strip() for x in txt.strip().split("-")]
    app = txt.split("\n")
    app = [x[1:].strip() if x[:1]=="-" else re.sub(r'^\d+\.', '', x).strip() for x in app]
    #app = [x[1:].strip() if x[:1]=="-" else x for x in app]
    app = [x for x in app if len(x)>0]
    return app

eps = 2
def match_string(lst,next):
    dists = [lev(x.lower(), next.lower()) for x in lst]
    #print(dists)
    pos_rank = np.array(dists).argmin()
    if dists[pos_rank] > eps:
        print("print eps", dists[pos_rank])
        #print(lst[pos_rank], "|", next)
        print(lst[:10], "|", next)
        return np.inf
    else: 
        return pos_rank

def rescale_rank(pos_rank,rank_len,eval_neg_items):
    return pos_rank*eval_neg_items/rank_len

def compute_ndcg(ranks, metric_at):
    app_dcg, app_hr = 0,0
    for rank in ranks:
        if rank < metric_at: #at 10
            app_dcg += 1 / np.log2(rank + 2)
            app_hr += 1
    app_idcg = np.sum([1 / np.log2(r + 2) for r in range(min(len(ranks),metric_at))]) #/IDCG
    ndcg = app_dcg / app_idcg
    return ndcg

def compute_hr(ranks, metric_at) -> float:
    return sum([1 for rank in ranks if rank < metric_at]) / len(ranks)

eval_neg_items = 100
metric_at = 10

exp_id: str = "ml-1m_10"

filename = f"../data/processed/{exp_id}_chatgpt.json"
out = read_json(filename)

data: dict = read_json(f"../data/processed/{exp_id}.json")

parsed_out = {}
for user_id, txt in out.items():
    parsed_out[user_id] = parse_gpt_output(txt)

rank_vec = []
tot_recall = []
tot_ndcg = []
tot_rls_rbo = []
tot_rls_jac = []
dictionary = [(int(key), value) for key, value in parsed_out.items()]
dictionary.sort(key=lambda x: x[0])
for counter, (user_id,user_out) in enumerate(dictionary, start=1):
    user_id = str(user_id)

    #print("UID",user_id)
    user_next = data[user_id]["str_v"]["next"]
    #print("NXT:",user_next)
    #print("OUT:",user_out)

    all_movies = user_out + user_next
    movie_to_id = dict(zip(all_movies,range(len(all_movies))))

    user_out_as_ints = [movie_to_id[i] for i in user_out]
    user_next_as_ints = [movie_to_id[i] for i in user_next]

    # print("P:", user_out_as_ints)
    # print("G:", user_next_as_ints)

    rls_rbo, rls_jac = compute_single_rls(user_out_as_ints[:10],user_next_as_ints,metrics_at=[metric_at], rbo_p=0.9)
    tot_rls_rbo.append(rls_rbo)
    tot_rls_jac.append(rls_jac)

    #print("rls_rbo", rls_rbo, "rls_jac", rls_jac)

    ranks = []
    for item_str in user_next:
        # print("user out", user_out)
        # exit(0)
        ranks.append(match_string(user_out, item_str))
        #print(pos_rank)

    # app_dcg, app_hr = 0,0
    # for rank in ranks:
    #     if rank < metric_at: #at 10
    #         app_dcg += 1 / np.log2(rank + 2)
    #         app_hr += 1
    

 
        


    # app_idcg = np.sum([1 / np.log2(r + 2) for r in range(min(len(ranks),metric_at))]) #/IDCG
    tot_ndcg.append(compute_ndcg(ranks, 10))

    tot_recall.append(compute_hr(ranks, 10))

    # app_rank_vec.append(pos_rank/len(user_out))

    # #pos_rank = rescale_rank(pos_rank,len(user_out),eval_neg_items)

    # pos_rank += 1 #rank should start at 1

    # print()

    # if pos_rank<=metric_at:
    #     app_tot_ndcg.append(1/np.log2(pos_rank+1))
    #     app_tot_recall.append(1)
    # else:
    #     app_tot_ndcg.append(0)
    #     app_tot_recall.append(0)
        
mean_ndcg_at = np.mean(tot_ndcg)
mean_recall_at = np.mean(tot_recall)
mean_rls_rbo_at = np.mean(tot_rls_rbo)
mean_rls_jac_at = np.mean(tot_rls_jac)
print("rank vec", rank_vec)
print("ncdg vec", tot_ndcg)
print("mean_ndcg_at",metric_at,"rescaled for",eval_neg_items,"negative items:",mean_ndcg_at)
print("mean_recall_at",metric_at,"rescaled for",eval_neg_items,"negative items:",mean_recall_at)
print("mean_rls_rbo_at",metric_at,"rescaled for",eval_neg_items,"negative items:",mean_rls_rbo_at)
print("mean_rls_jac_at",metric_at,"rescaled for",eval_neg_items,"negative items:",mean_rls_jac_at)
