#!/bin/bash

echo "facebook/bart-large-mnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli"

echo "joeddav/xlm-roberta-large-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli"

echo "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

echo "BaptisteDoyen/camembert-base-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli"