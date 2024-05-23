#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt.py electra_hypopt mercelisw/electra-grc greek_glaux
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt.py AG_BERT_hypopt pranaydeeps/Ancient-Greek-BERT NFC
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt.py greberta_hypopt bowphs/GreBerta NFC
CUDA_VISIBLE_DEVICES=2 python run_w_hypopt.py UGARIT_hypopt UGARIT/grc-alignment NFKC