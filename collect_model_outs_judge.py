from datasets import load_dataset
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time

import torch
# from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
# from inference.generate import SpeculativeGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="psoups")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--run_percent", type=float, default=100.)
parser.add_argument("--target", type=str, default='')
parser.add_argument("--draft", type=str, default='')
parser.add_argument("--max_new_token", type=int, default=128)

parser.add_argument("--draft_gpu", type=str, default="cuda:0")
parser.add_argument("--target_gpu", type=str, default="cuda:1")
parser.add_argument("--recover", action='store_true', default = False)

parser.add_argument("--config", type=str, default='configs/hh_llama3.config')

parser.add_argument("--out_file", type=str, default='')

parser.add_argument("--sys_prompt", type=str, default="")
args = parser.parse_args()

from argsearch_judge import ARGS_JUDGE

print(f"{args=}")

if args.recover:
    print("[INFO]: LOOKS LIKE YOU WANT TO RECOVER SOME RESULTS,")
    print("[INFO]: MAKE SURE ALL COMMANDLINE ARGS ARE EXACTLY THE SAME!!!")
    input("PRESS ENTER TO CONTINUE")

if not (args.max_new_token > 0):
    print("ERROR: Max tokens should be greater than 0!")
    exit(1)

cfg_path = Path(args.config)
if not cfg_path.exists():
    print("ERROR: Config doesn't exist!")
    exit(1)
    
out_path = Path(args.out_file + f"_0.jsonl")
if out_path.exists() and (not args.recover):
    print("ERROR: out_path already exists!")
    exit(1)

if not out_path.exists() and args.recover:
    print("ERROR: out_path DOESN'T exist!")
    exit(1)

with open(cfg_path) as f:
    run_configs = [json.loads(line) for line in f.readlines()]
    
# validate configs
for run_config in run_configs:
    if "rm_weight" not in run_config:
        print(f"Missing key 'rm_weight' in {run_config=}")
        exit(1)
    elif "topk" not in run_config:
        print(f"Missing key 'topk' in {run_config=}")
        exit(1)
    elif "mode" not in run_config:
        print(f"Missing key 'mode' in {run_config=}")
        exit(1)
    elif "sample_temp" not in run_config:
        print(f"Missing key 'sample_temp' in {run_config=}")
        exit(1)
    elif "perspective" not in run_config:
        run_config["perspective"]=None

# print(f"[INFO]: Loaded {len(run_configs)} run configs.")
# print(f"[DEBUG]: {run_configs=}")
    
print(f"[INFO]: Loading dataset ({args.dataset=}, {args.split=})")

if args.dataset == "psoups":
    import json
    
    file_path = 'koala_eval_50.json'

    
    prompts = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
        for item in data:
            prompts.append(item['prompt'])  
    # print(prompts)
    test_ds = prompts
elif args.dataset == "steers":
    import json
    # 路径到你的JSON文件
    test_ds = []
    file_path = '/home/xiaotian/compared_experiments/data/nvidia/HelpSteer2/validation.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file) 
        for item in data:
            if len(item["prompt"]) <= 100:
                test_ds.append(item["prompt"])
    
    test_ds = test_ds[1::2]
    
end_idx = int(len(test_ds) * (args.run_percent/100.))
print(f"[INFO]: {end_idx=}, {len(test_ds)=}")

truncated_ds = test_ds[0:end_idx]
print(f"{len(truncated_ds)=}")

print(f"[INFO]: Loading models ({args.draft=}, {args.target=})")
search = ARGS_JUDGE(draft_path=args.draft, target_path=args.target, draft_dev=args.draft_gpu, target_dev=args.target_gpu)
print(f"[INFO]: Done")

hh_rlhf_all_aspects = {'harmlessness': 'Harmlessness: The response should avoid content that is offensive, discriminatory, or harmful',
               'helpfulness': 'Helpfulness: The response should provide useful resources and suggestions to the user',
               'humour': 'Humour: The response should be cheerful and amusing'}

def runprompt(base_prompt: str, rm_weight=0., topk=5, new_token=128, mode="p_sigmoid_mixing", sample_temp=None, draft_dev:str="cuda:0", perspective=None) -> str:
    # preference_prompt = 'The response should provide useful information or solutions that help the user understand the issue or make a decision. It should be practical, clear, and specific. Meanwhile, the response should avoid any content that could be misunderstood or hurt someone\'s feelings. Please remain neutral and respectful, avoiding negative language or offensive comments. Moreover, The response should be light, concise, like an old friend but still maintain professionalism.'
    preference_prompt = ''
    prompt = preference_prompt + base_prompt
    # hhh
    # draft_prompt = "The response should avoid content that is offensive, discriminatory, or harmful. Meanwhile, it should be cheerful and amusing."
    # target_prompt = "The response should provide useful resources and suggestions to the user."

    # ultra-feedback
    # draft_prompt = "The response should actively making known all the full truth of a matter, meanwhile, it should provide useful resources and suggestions to the user."
    # target_prompt = "The response should follow the instructions of the query, and it should not tell lies."

    # imhi   
    # draft_prompt = "The response should provide evidence with high quality and reliability, and it should express clear logic and provide consistent evidencethe."
    # target_prompt = "The explanations should make correct predictions."

    # vivid only
    # draft_prompt = "The response should be richly descriptive, with clarity and detail."
    # target_prompt = "The response should bring ideas to life through descriptive language, sensory details, and a strong sense of presence."

    # vivid and creative
    draft_prompt = "The response should be richly descriptive, with clarity and detail."
    target_prompt = "The response should be imaginative, and showcases innovative ideas"

    # other
    # draft_prompt = "The response should be emotionally resonant, capable of evoking empathy or deep feelings, meanwhile, it should be richly descriptive, with clarity and detail."
    # target_prompt = "The response should be imaginative, and showcases innovative ideas"

    # single-harmless
    # draft_prompt = "The response should avoid any content that could be misunderstood or hurt someone\'s feelings. Please remain neutral and respectful, avoiding negative language or offensive comments."
    # target_prompt = "The response should avoid content that is offensive, discriminatory, or harmful."

    # single-helpful
    # draft_prompt = "The response should be professional, only a PhD Student in that specific field could understand."
    # target_prompt = "The response should provide useful resources and suggestions to the user."

    # hh
    # draft_prompt = "The response should avoid any content that could be misunderstood or hurt someone\'s feelings. Please remain neutral and respectful, avoiding negative language or offensive comments."
    # target_prompt = "The response should provide useful resources and suggestions to the user, which only a PhD Student in that specific field could understand."


    # harmless_prompt ="The response should avoid any content that could be misunderstood or hurt someone\'s feelings. Please remain neutral and respectful, avoiding negative language or offensive comments." #harmless
    # humor_prompt ="The response should be light, concise, like an old friend in a humorous style but still maintain professionalism."  #humor
    # helpful_prompt ="The response should provide useful resources and suggestions to the user, which only a PhD Student in that specific field could understand." #helpful
    if perspective == None:
        tokens = search.generate(draft_prompt, target_prompt, prompt, method=mode, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False)
    else:
        tokens = search.generate(draft_prompt, target_prompt, prompt, topk=topk, max_new_token=new_token, weight=rm_weight, debug=False, perspective=perspective)

    # too long seqlen
    if tokens == None: return None, None
    raw_tokens = tokens[0].detach().cpu().numpy().tolist()
    tokens_text = search.tokens_to_text(tokens)[0]
    cleaned_text = tokens_text.replace(preference_prompt, '')
    del tokens
    tokens_text_np = cleaned_text.removeprefix(prompt)
    return tokens_text_np, raw_tokens

for config_num, run_config in enumerate(run_configs):
    print(f"[INFO]: Running config: {run_config=}")

    data = []
    if args.recover and Path(args.out_file + f"_{config_num}.jsonl").exists():
        continue
        print(f"[INFO]: Run already exists, checking if it's done")
        resfile = open(Path(args.out_file + f"_{config_num}.jsonl"))
        samples = resfile.readlines()


        last_obj = json.loads(samples[-2])
        if last_obj["prompt"] != truncated_ds[len(samples) -1]:
            print(f"[INFO]: PROMPTS DID NOT MATCH RECOVERY FAILED!!!")
            exit(1)

    for idx, ds_row in enumerate(tqdm(truncated_ds)):
        # if args.recover and (idx <= len(samples) -1):
        #     print(f"[INFO]: SKIPPING {idx}")
        #     continue

        # print(f"{ds_row=}")
        current_prompt = args.sys_prompt + ds_row #["prompt"]
        start = time.time()
        res, tokens = runprompt(current_prompt, float(run_config["rm_weight"]), run_config["topk"], args.max_new_token, run_config["mode"], run_config["sample_temp"], perspective=run_config["perspective"], draft_dev=args.draft_gpu)
        if tokens == None:
            print("Too long, skipped")
            continue

        elapsed = time.time() -start

        data.append({"prompt": current_prompt, "result": res, "response": current_prompt + res, "elapsed":elapsed, "method": args.out_file + f"_{config_num}"})
        # print(f"[DEBUG]: {elapsed=} {len(current_prompt)=} {current_prompt=}, {res=}")
        with open(Path(args.out_file + f"_{config_num}.jsonl"), "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False)
