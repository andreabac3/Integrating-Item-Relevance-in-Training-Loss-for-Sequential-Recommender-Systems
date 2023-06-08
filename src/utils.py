
import json
import numpy as np


def read_json(filename: str) -> dict:
    with open(filename) as reader:
        return json.load(reader)

def write_json(filename: str, data: dict) -> None:
    with open(filename, 'w') as writer:
        json.dump(data, writer, indent=4, sort_keys=True)

def compute_rls_metrics(preds1, preds2, metrics_at=np.array([1,5,10,20,50]), rbo_p=0.9):
    rls_rbo = np.zeros((len(metrics_at)))
    rls_jac = np.zeros((len(metrics_at)))
    
    for pred1,pred2 in zip(preds1,preds2):
        app_rbo, app_jac = compute_single_rls(pred1,pred2,metrics_at=np.array([1,5,10,20,50]),rbo_p=0.9)
        rls_rbo += app_rbo
        rls_jac += app_jac
                
    rbo_dict = {"@"+str(k):rls_rbo[i]/len(preds1) for i,k in enumerate(metrics_at)}
    jac_dict = {"@"+str(k):rls_jac[i]/len(preds1) for i,k in enumerate(metrics_at)}
    
    return {"RLS_RBO":rbo_dict, "RLS_JAC":jac_dict}

def compute_single_rls(pred1,pred2,metrics_at=np.array([1,5,10,20,50]),rbo_p=0.9):
    rls_rbo = np.zeros((len(metrics_at)))
    rls_jac = np.zeros((len(metrics_at)))

    j = 0
    rbo_sum = 0
    for d in range(1,min(min(len(pred1),len(pred2)),max(metrics_at))+1):
        set_pred1, set_pred2 = set(pred1[:d]), set(pred2[:d])
        inters_card = len(set_pred1.intersection(set_pred2))
        rbo_sum += rbo_p**(d-1)*inters_card/d
        
        if d==metrics_at[j]:
            rls_rbo[j] += (1-rbo_p)*rbo_sum
            rls_jac[j] += inters_card/len(set_pred1.union(set_pred2))
            j+=1
            
    #Check if it has stopped before cause pred1 or pred2 are shorter
    if j!=len(metrics_at):
        for k in range(j,len(metrics_at)):
            rls_rbo[k] += (1-rbo_p)*rbo_sum
            rls_jac[k] += inters_card/len(set_pred1.union(set_pred2))

    return rls_rbo, rls_jac