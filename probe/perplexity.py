from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random
import numpy as np
import json


def set_random_seeds(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds()

MODEL_PATH = ''
test_data_path = f''
save_path = './test_output'
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
).eval()

Perplexity_dict = {}

for idx, item in tqdm(enumerate(test_data, 1)):
    # Modify the format of your own dataset
    instruction = item['instruction']
    input = item.get('input', None)
    output = item['output']
    if input:
        instruction = instruction + '\n' + input

    instruction_ids_len = tokenizer.encode(instruction, return_tensors="pt").shape[1]
    generated_ids = tokenizer(instruction + output, return_tensors="pt")
    generated_ids = generated_ids.input_ids.to(model.device)

    max_length = 32768
    stride = 1
    seq_len = generated_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = generated_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if begin_loc == 0:
            target_ids[:, :instruction_ids_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()

        if num_valid_tokens > 0:
            nll_sum += neg_log_likelihood * num_valid_tokens
            n_tokens += num_valid_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)

    Perplexity_dict[str(idx)] = [{'test_sentence': output, 'ppl': ppl.item()}]
    with open(save_path + 'Perplexity_dict.json', 'w', encoding='utf-8') as f:
        json.dump(Perplexity_dict, f, ensure_ascii=False, indent=2)


