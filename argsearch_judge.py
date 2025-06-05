from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
from LLaMAFactory.src.llamafactory.model.loader import load_model
from LLaMAFactory.src.llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
from transformers import Seq2SeqTrainingArguments
from peft import get_peft_model, TaskType, LoraConfig
data_args = DataArguments
training_args = Seq2SeqTrainingArguments
finetuning_args = FinetuningArguments

from sampling import autoregressive_sampling, base_speculative_sampling, trans_speculative_sampling

# import the huggingface transformers libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
#### auto size stuff
import numpy as np


# from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
# from inference.generate import SpeculativeGenerator


class ARGS_JUDGE:
    def __init__(self, draft_path, target_path, draft_dev="cuda:0", target_dev="cuda:1", torch_dtype=torch.float16):
        self.draft_dev = draft_dev
        self.draft_path = draft_path
        self.target_dev = target_dev
        print("Loading Draft Model...")

        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_path, torch_dtype=torch_dtype).to(self.draft_dev)
        self.model_args = ModelArguments
        self.model_args.model_name_or_path = target_path

        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(draft_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.draft_model.pad_token = self.tokenizer.eos_token

        print("Loading Judge Model...")

        self.target_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch_dtype).to(self.target_dev)

    def get_input_ids(self, draft_prompt: str, target_prompt: str, prompt: str) -> torch.Tensor:
        messages = [
            {"role": "user", "content": prompt},
        ]
        draft_role = [
            {"role": "user", "content": draft_prompt},
        ]
        target_role = [
            {"role": "user", "content": target_prompt},
        ]
        # tokens = self.tokenizer(messages, return_tensors="pt").input_ids.to(self.llm_dev)
        tokens_draft = self.tokenizer.apply_chat_template(draft_role, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.draft_dev)
        tokens_target = self.tokenizer.apply_chat_template(target_role, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.draft_dev)
        tokens_message = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.draft_dev)
        # print(tokens)
        return  tokens_draft, tokens_target, tokens_message
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]: 
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def generate(self, draft_prompt, target_prompt, prompt, weight=0., topk=1, max_new_token=128, method="greedy", num_tokens=128, temperature=0.9, chunk_size=5, debug=False, random_seed=None):

        tokens_draft, tokens_target, tokens = self.get_input_ids(draft_prompt, target_prompt, prompt)
        # print(f"data type:{type(tokens)}")

        top_k = 20
        top_p = 0.9
        
        torch.manual_seed(123)
        
        if tokens.shape[-1] > self.draft_model.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
        
        if tokens.shape[-1] > self.target_model.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
          
        rm_cached = None
        cached = None

        # base judge
        # output_token = base_speculative_sampling(tokens, tokens_draft, tokens_target, self.draft_model, self.target_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, dev_draft=self.draft_dev, dev_target=self.target_dev, temperature=temperature)
        # Persona-judge
        output_token = trans_speculative_sampling(tokens, tokens_draft, tokens_target, self.draft_model, self.target_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed, dev_draft=self.draft_dev, dev_target=self.target_dev, temperature=temperature)
        print("finished one speculative_sampling")
                
        return output_token