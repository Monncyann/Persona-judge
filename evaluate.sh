#!/bin/sh

#helpful
python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx.jsonl" \
    --tokenizer="ArmoRM-Llama3-8B-v0.1" \
    --rm="ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:2" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7\
    --dimension=0

python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx.jsonl" \
    --tokenizer="gpt2-large-helpful-reward_model" \
    --rm="gpt2-large-helpful-reward_model" \
    --rm_gpu="cuda:2" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7
#harmless
python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx.jsonl" \
    --tokenizer="ArmoRM-Llama3-8B-v0.1" \
    --rm="ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:2" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7\
    --dimension=10

python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx.jsonl" \
    --tokenizer="gpt2-large-harmless-reward_model" \
    --rm="gpt2-large-harmless-reward_model" \
    --rm_gpu="cuda:2" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7