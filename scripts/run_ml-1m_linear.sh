







device="cuda:1"

dataset_name="ml-1m"


eval_positive="10"

loss_type="exp"

experiment_id="${dataset_name}_${eval_positive}"


mkdir -p ../out/experiments_our_log/${experiment_id}/


# Multi positive + Relevance Loss
python3 sasrec_main.py --experiment_id ${experiment_id} --train_num_positives 2 --train_num_negatives 2 --device ${device} --loss_type ${loss_type} >> ../out/experiments_our_log/${experiment_id}/multi_positive_2_${loss_type}.txt &
python3 sasrec_main.py --experiment_id ${experiment_id} --train_num_positives 3 --train_num_negatives 3 --device ${device} --loss_type ${loss_type} >> ../out/experiments_our_log/${experiment_id}/multi_positive_3_${loss_type}.txt &
python3 sasrec_main.py --experiment_id ${experiment_id} --train_num_positives 4 --train_num_negatives 4 --device ${device} --loss_type ${loss_type} >> ../out/experiments_our_log/${experiment_id}/multi_positive_4_${loss_type}.txt &
python3 sasrec_main.py --experiment_id ${experiment_id} --train_num_positives 5 --train_num_negatives 5 --device ${device} --loss_type ${loss_type} >> ../out/experiments_our_log/${experiment_id}/multi_positive_5_${loss_type}.txt &
python3 sasrec_main.py --experiment_id ${experiment_id} --train_num_positives 10 --train_num_negatives 10 --device ${device} --loss_type ${loss_type} >> ../out/experiments_our_log/${experiment_id}/multi_positive_10_${loss_type}.txt &




