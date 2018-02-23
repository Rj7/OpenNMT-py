#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-07:59
#SBATCH --output=base_%N-%j.out
#module load miniconda3
source activate pytorch
python preprocess.py -train_src /home/rajaunbc/project/rajaunbc/data/stanfordnmt/training.tok.en -train_tgt /home/rajaunbc/project/rajaunbc/data/stanfordnmt/training.tok.de -valid_src /home/rajaunbc/project/rajaunbc/data/stanfordnmt/newstest2015.tok.en -valid_tgt /home/rajaunbc/project/rajaunbc/data/stanfordnmt/newstest2015.tok.de -save_data data/stan_en_de_50k -src_vocab_size 50000 -tgt_vocab_size 50000 -lower
python train.py -data data/stan_en_de_50k -save_model models/stan_en_de_50k_baseline -gpuid 0 -encoder_type 'brnn' -brnn_merge 'concat' -rnn_size 1000 -rnn_type 'GRU' -global_attention 'mlp' -optim 'adadelta' -batch_size 128 -epochs 13 -input_feed -max_grad_norm 1
#python translate.py -model models/bahdanau_en_de_acc_57.55_ppl_8.27_e5.pt -src data/newstest2013.tok.en -tgt data/newstest2013.tok.de -output pred.txt -replace_unk -verbose -report_bleu

