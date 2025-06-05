#!/bin/sh

python Persona-judge/collect_model_outs_judge.py\
    --target="Llama-3-Base-8B-SFT" \
    --draft="Llama-3-Base-8B-SFT" \
    --out_file="Persona-judge/results/trans_judge/xxxx" \
    --dataset="psoups" \
    --draft_gpu="cuda:0" \
    --target_gpu="cuda:1" \
    --run_percent=10.