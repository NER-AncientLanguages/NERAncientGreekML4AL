#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt_reduced.py deberta_hypopt_reduced /home/local/marijke/deberta/greek_small_cased_model greek_glaux --tokenizer_path /home/local/marijke/deberta/greek_small_cased_model/tokenizer
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt_reduced.py electra_hypopt_reduced mercelisw/electra-grc greek_glaux
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt_reduced.py AG_BERT_hypopt_reduced pranaydeeps/Ancient-Greek-BERT NFC
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt_reduced.py greberta_hypopt_reduced bowphs/GreBerta NFC
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt_reduced.py UGARIT_hypopt_reduced UGARIT/grc-alignment NFKC