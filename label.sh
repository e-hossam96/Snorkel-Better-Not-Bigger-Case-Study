#!/bin/bash

CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 0 --end_idx 1000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 1000 --end_idx 2000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 2000 --end_idx 3000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 3000 --end_idx 4000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 4000 --end_idx 5000
CUDA_VISIBLE_DEVICE=0 python3 label.py --model_name "facebook/bart-large-mnli" --start_idx 5000 --end_idx 6000

