# Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment
This repository contains the official implementation of the paper Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment (https://arxiv.org/abs/2504.12663). Persona-judge eliminates the need for external reward signals or policy fine-tuning. By leveraging the model’s inherent capability for preference judgment, Persona-judge effectively aligns multi-dimensional preferences in the prediction of the next token.

Acknowledgement: This repository is built based on https://github.com/deeplearning-wisc/args.

## Setup
The following packages, and versions were used:

```bash=
git clone https://github.com/Monncyann/Persona-judge.git

conda create -n pjudge python=3.9 -y
conda activate pjudge

cd Persona-judge
pip -r requirements.txt
```
## Data and Metrics
We conducted experiments using two datasets in total: nvidia/HelpSteer2/validation.json (https://huggingface.co/datasets/nvidia/HelpSteer2) and Personalized Soups koala_eval_50.json (https://github.com/joeljang/RLPHF). The corresponding hyperparameters can be modified via `--dataset="psoups"` and `--dataset="steers"`.

For reward models evaluation, we utilize RLHFlow/ArmoRM-Llama3-8B-v0.1 (https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1), gpt2-large-helpful-reward_model (https://huggingface.co/Ray2333/gpt2-large-helpful-reward_model) and gpt2-large-harmless-reward_model (https://huggingface.co/Ray2333/gpt2-large-harmless-reward_model). For GPT-as-a-judge, we call the GPT-4 via APIs.


## Training
Persona-judge is a training-free method, this project does not include the release of model checkpoints.

## Text generation with Persona-judge
After preparing the personalized prompt you needed, the following command in `collect_model_outs_judge.sh` can be run to start personalized generation:
```python
python Persona-judge/collect_model_outs_judge.py\
    --target="Llama-3-Base-8B-SFT" \
    --draft="Llama-3-Base-8B-SFT" \
    --out_file="Persona-judge/results/trans_judge/xxx" \
    --dataset="psoups" \
    --draft_gpu="cuda:0" \
    --target_gpu="cuda:1" \
    --run_percent=10.
```
For example, if you prefer the response to be vivid and creative, you could use the following prompts to generate:
```python
# vivid and creative
    draft_prompt = "The response should be richly descriptive, with clarity and detail."
    target_prompt = "The response should be imaginative, and showcases innovative ideas."
```
The choice of which preference performs the draft or judge model does not significantly affect the outcome.

## Evaluation
To run the evaluations, execute the following commands in the `evaluate.sh`:
```python
#helpful
python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx" \
    --tokenizer="ArmoRM-Llama3-8B-v0.1" \
    --rm="ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:0" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7\
    --dimension=0

python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx" \
    --tokenizer="gpt2-large-helpful-reward_model" \
    --rm="gpt2-large-helpful-reward_model" \
    --rm_gpu="cuda:0" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7
    
#harmless
python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx" \
    --tokenizer="ArmoRM-Llama3-8B-v0.1" \
    --rm="ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:0" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7\
    --dimension=10

python Persona-judge/measure_reward.py \
    --out_file="Persona-judge/results/trans_judge/xxx" \
    --tokenizer="gpt2-large-harmless-reward_model" \
    --rm="gpt2-large-harmless-reward_model" \
    --rm_gpu="cuda:0" \
    --experiment="hhrlhf" \
    --start_index=0\
    --end_index=7
```

## Citation

If you find this repository useful in your research, please consider citing:

```
@article{zhang2025persona,
  title={Persona-judge: Personalized Alignment of Large Language Models via Token-level Self-judgment},
  author={Zhang, Xiaotian and Chen, Ruizhe and Feng, Yang and Liu, Zuozhu},
  journal={arXiv preprint arXiv:2504.12663},
  year={2025}
}
```
