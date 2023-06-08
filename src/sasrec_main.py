import os
import time
from typing import Optional
import torch
import argparse

from sasrec_model import SASRec
from sasrec_utils import *


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
parser.add_argument('--num_epochs', default=11, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

parser.add_argument('--train_num_positives', default=1, type=int)
parser.add_argument('--train_num_negatives', default=1, type=int)
parser.add_argument('--loss_type', default="fixed", type=str, required=False)


args = parser.parse_args()
seed_everything(1234)
if "ml-1m" in args.experiment_id.lower() and args.maxlen != 200:
    print(f"WARNING: Dataset ml-1m you should use maxlen of 200 instead of {args.maxlen}")


USE_RELEVANCE_LOSS: bool = args.loss_type != "fixed"
assert args.loss_type in ["fixed", "linear", "exp", "pow2"], f"Error: the following loss_type is not supported: {args.loss_type}"
print("use_relevance_loss", USE_RELEVANCE_LOSS)

log_folder = os.path.join(args.train_dir,"out","log")
if not os.path.isdir(log_folder):
    os.makedirs(log_folder)
model_folder = os.path.join(args.train_dir,"out","model")
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)
dump_metrics_folder = os.path.join(args.train_dir,"out","dump_metrics")
if not os.path.isdir(dump_metrics_folder):
    os.makedirs(dump_metrics_folder)
# with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

if __name__ == '__main__':
    # global dataset
    dataset = load_pickle_dataset(args.train_dir,args.experiment_id)

    usernum, itemnum = len(dataset["umap"]), len(dataset["smap"])

    #remove time
    for key in ["train","val","test"]:
        dataset[key] = {k:v[0] for k,v in dataset[key].items()}
    user_train,user_valid,user_test = [dataset[key] for key in ["train","val","test"]]

    #[user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    f = open(os.path.join(log_folder, f'{args.experiment_id}.txt'), 'w') #+ '_' + args.train_dir

    #sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    sampler = WarpSampler_FS(user_train, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3, num_positives=args.train_num_positives, num_negatives=args.train_num_positives)
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace() 
    #valid_user 6040
    
    epoch_exp_unique_name = "SASRec.expid={}.epoch={}.npositive={}.nnegative={}.losstype={}".format(args.experiment_id, 0, args.train_num_positives, args.train_num_negatives, args.loss_type)
    model.eval()
    t_test = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args, use_test=True, metrics_loc=os.path.join(dump_metrics_folder, epoch_exp_unique_name+".test.json"))
    t_valid = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args, use_test=False, metrics_loc=os.path.join(dump_metrics_folder, epoch_exp_unique_name+".valid.json"))
    # print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f, RLS-RBO %.4f, RLS-JAC %.4f, NDCG@5 %.4f,  NDCG@15 %.4f,  NDCG@20 %.4f, HR@5 %.4f, HR@15 %.4f, HR@20 %.4f)'
    #       % (epoch, T, t_valid.ndcg, t_valid.hr, t_test.ndcg, t_test.hr, t_test.rls_rbo, t_test.rls_jacc, t_test.ndcg_5, t_test.ndcg_15, t_test.ndcg_20, t_test.hr_5, t_test.hr_15, t_test.hr_20))
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    bce_criterion_pos = torch.nn.BCEWithLogitsLoss(reduction='none') # torch.nn.BCELoss()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    T = 0.0
    t0 = time.time()

    num_pos = args.train_num_positives
    
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break
        for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            #u, seq, pos, neg = torch.tensor(u), torch.tensor(seq), torch.tensor(pos), torch.tensor(neg)

            pos_logits, neg_logits = model(u, seq, pos, neg)

            # pos_logits.shape == (num_sample, max_len, num_positivies) --> pos sono da quello piu' vicino temporalmente a quello piu' lontano.
            
            # creo una matrice con la stessa shape rescale_vec2 = torch.ones(pos_logits.shape) * rescale_vec 
            #rescale_vec2 shape: (batch_size, seq_len, num_positives)

            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)

            #loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            
            if USE_RELEVANCE_LOSS:
                pos_loss = bce_criterion_pos(pos_logits[indices], pos_labels[indices])
                rescale_vec = torch.arange(num_pos, 0, -1, device=args.device, dtype=torch.float)
                if args.loss_type == "exp":
                    rescale_vec = torch.exp(rescale_vec)
                elif args.loss_type == "pow2":
                    rescale_vec **= 2
                else:
                    assert args.loss_type == "linear"
                rescale_vec /= (num_pos*(num_pos+1)/2)
                rescale_vec = torch.ones(pos_logits.shape,device=args.device) * rescale_vec
                scaled_loss = rescale_vec[indices] * pos_loss
                loss = scaled_loss.mean()
            else:
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            #print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        epoch_exp_unique_name = "SASRec.expid={}.epoch={}.npositive={}.nnegative={}.losstype={}".format(args.experiment_id,epoch, args.train_num_positives, args.train_num_negatives, args.loss_type)
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args, use_test=True, metrics_loc=os.path.join(dump_metrics_folder, epoch_exp_unique_name+".test.json"))
            t_valid = evaluate(model, user_train, user_valid, user_test, usernum, itemnum, args, use_test=False, metrics_loc=os.path.join(dump_metrics_folder, epoch_exp_unique_name+".valid.json"))
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, RLS-RBO %.4f, RLS-JAC %.4f), test (NDCG@10: %.4f, HR@10: %.4f, RLS-RBO %.4f, RLS-JAC %.4f, NDCG@5 %.4f,  NDCG@15 %.4f,  NDCG@20 %.4f, HR@5 %.4f, HR@15 %.4f, HR@20 %.4f, NDCG@5 %.4f, NDCG@10_REL: %.4f, NDCG@15_REL %.4f,  NDCG@20_REL %.4f)'
                    % (epoch, T, t_valid.ndcg_10, t_valid.hr_10, t_valid.rls_rbo, t_valid.rls_jac, t_test.ndcg_10, t_test.hr_10, t_test.rls_rbo, t_test.rls_jac, t_test.ndcg_5, t_test.ndcg_15, t_test.ndcg_20, t_test.hr_5, t_test.hr_15, t_test.hr_20, t_test.ndcg_5_rel, t_test.ndcg_10_rel, t_test.ndcg_15_rel, t_test.ndcg_20_rel))            
            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            fname = f'{epoch_exp_unique_name}.pth'
            torch.save(model.state_dict(), os.path.join(model_folder, fname))
    
    f.close()
    sampler.close()
    print("Done")
