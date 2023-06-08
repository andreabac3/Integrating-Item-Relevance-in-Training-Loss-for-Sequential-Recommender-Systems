import os
import time
import torch
import argparse

from sasrec_model import SASRec
from sasrec_utils import *


def str2bool(s):
    assert s.lower() in ['false', 'true'], f"str2bool(): {s} is not a valid boolean string"
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', required=True)
parser.add_argument('--train_dir', default="../")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int, help="History len")
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--use_relevance_loss', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--train_num_positives', default=1, type=int)
parser.add_argument('--train_num_negatives', default=1, type=int)





if __name__ == '__main__':
    args = parser.parse_args()
    # global dataset
    dataset = load_pickle_dataset(args.train_dir, args.experiment_id)
    usernum, itemnum = len(dataset["umap"]), len(dataset["smap"])
    for key in ["train","val","test"]:
        dataset[key] = {k:v[0] for k,v in dataset[key].items()}
    user_train,user_valid,user_test = [dataset[key] for key in ["train","val","test"]]

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    log_folder = os.path.join(args.train_dir,"out","log")
    
    f = open(os.path.join(log_folder, f'{args.experiment_id}.txt'), 'w') #+ '_' + args.train_dir


    sampler = WarpSampler_FS(user_train, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, num_positives=args.train_num_positives, num_negatives=args.train_num_positives)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

    model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))

    model.eval()

    print("Evaluating ...")
    t_test = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args, use_test=True)

    print('test (NDCG@10: %.4f, HR@10: %.4f, RLS-RBO %.4f, RLS-JAC %.4f, NDCG@5 %.4f,  NDCG@15 %.4f,  NDCG@20 %.4f, HR@5 %.4f, HR@15 %.4f, HR@20 %.4f)' % (t_test.ndcg, t_test.hr, t_test.rls_rbo, t_test.rls_jacc, t_test.ndcg_5, t_test.ndcg_15, t_test.ndcg_20, t_test.hr, t_test.hr_5, t_test.hr_15, t_test.hr_20))