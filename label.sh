#!/bin/bash

echo "facebook/bart-large-mnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 0 --end_idx 500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 500 --end_idx 1000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 1000 --end_idx 1500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 1500 --end_idx 2000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 2000 --end_idx 2500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 2500 --end_idx 3000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 3000 --end_idx 3500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 3500 --end_idx 4000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 4000 --end_idx 4500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 4500 --end_idx 5000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 5000 --end_idx 5500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 5500 --end_idx 6000

echo "joeddav/xlm-roberta-large-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 0 --end_idx 500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 500 --end_idx 1000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 1000 --end_idx 1500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 1500 --end_idx 2000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 2000 --end_idx 2500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 2500 --end_idx 3000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 3000 --end_idx 3500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 3500 --end_idx 4000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 4000 --end_idx 4500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 4500 --end_idx 5000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 5000 --end_idx 5500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "joeddav/xlm-roberta-large-xnli" --start_idx 5500 --end_idx 6000

echo "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 0 --end_idx 500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 500 --end_idx 1000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 1000 --end_idx 1500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 1500 --end_idx 2000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 2000 --end_idx 2500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 2500 --end_idx 3000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 3000 --end_idx 3500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 3500 --end_idx 4000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 4000 --end_idx 4500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 4500 --end_idx 5000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 5000 --end_idx 5500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli" --start_idx 5500 --end_idx 6000

echo "BaptisteDoyen/camembert-base-xnli"
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 0 --end_idx 500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 500 --end_idx 1000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 1000 --end_idx 1500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 1500 --end_idx 2000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 2000 --end_idx 2500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 2500 --end_idx 3000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 3000 --end_idx 3500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 3500 --end_idx 4000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 4000 --end_idx 4500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 4500 --end_idx 5000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 5000 --end_idx 5500
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "BaptisteDoyen/camembert-base-xnli" --start_idx 5500 --end_idx 6000