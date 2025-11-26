# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


model_name = "Qwen/Qwen3-0.6B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

q_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": '求解sinx再x=3处的泰勒三阶展开式，并将最后答案放在{}中'}
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

base_input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'].to(DEVICE)
base_attention_mask = torch.ones_like(base_input_ids, device=DEVICE)

max_new_tokens = 4096
temperature = 0.6
top_k_sampling = 20
top_p = 0.95
eos = tokenizer.eos_token_id

N_warmup = 8  # 离线 warmup traces 数量。设为 0 跳过 warmup（用 manual_threshold）
eta = 90  # 保留 top-eta% traces，阈值 s = percentile(100-eta)
group_size = 1024  # 滑窗大小
k_conf = 10  # token confidence 使用 top-k logprobs 的平均（k_conf >=1）
num_traces = 8  # 主生成时要生成的 trace 数
manual_threshold = None


def generate_trace_with_conf(model, base_input_ids, max_new_tokens, stop_threshold=None):
    all_confs = torch.zeros(max_new_tokens, device=DEVICE)
    generated_tokens = torch.zeros(max_new_tokens, dtype=torch.long, device=DEVICE)
    
    past_key_values = None
    current_input = base_input_ids.clone()
    
    CompleteTrajectory = True
    actual_length = 0

    window_sum = 0.0
    min_gconf = float('inf')
    
    for step in tqdm(range(max_new_tokens), desc='Generate Up Stage'):
        with torch.no_grad():
            out = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]  # shape (1, V)
            past_key_values = out.past_key_values

        log_probs = F.log_softmax(logits, dim=-1)
        topk_vals, _ = torch.topk(log_probs, k=k_conf, dim=-1)
        Ci = -topk_vals.mean().item()
        all_confs[step] = Ci

        window_sum += Ci
        if group_size > 0 and step >= group_size:
            window_sum -= all_confs[step - group_size].item()
            gc = window_sum / group_size
        else:
            gc = window_sum / (step + 1)

        if gc < min_gconf:
            min_gconf = gc

        if temperature == 0.0:
            idx_next = torch.topk(logits, k=1, dim=-1)[1]
        else:
            lp = logits / temperature
            k_samp = top_k_sampling if top_k_sampling is not None else 0
            if k_samp > 0 or top_p < 1.0:
                lp = top_k_top_p_filtering(lp.clone(), top_k=k_samp, top_p=top_p)
            probs = F.softmax(lp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        
        next_token_id = idx_next[0, 0].item()

        if (stop_threshold is not None) and (gc < stop_threshold):
            CompleteTrajectory = False
            break

        generated_tokens[actual_length] = next_token_id
        actual_length += 1
        current_input = idx_next
        if next_token_id == eos:
            break

    if actual_length > 0:
        input_ids = torch.cat([
            base_input_ids,
            generated_tokens[:actual_length].unsqueeze(0)
        ], dim=1)
    else:
        input_ids = base_input_ids.clone()

    conf_list = all_confs[:actual_length].tolist() if actual_length > 0 else []
    
    return input_ids, conf_list, min_gconf, CompleteTrajectory


def generate_traces_batched(model, base_input_ids, max_new_tokens, num_traces):
    batch_size = num_traces
    input_ids = base_input_ids.repeat(batch_size, 1)

    all_confs = torch.zeros(max_new_tokens, batch_size, device=DEVICE)

    unfinished = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)
    finished_at_step = torch.full((batch_size,), max_new_tokens, dtype=torch.long, device=DEVICE)

    past_key_values = None
    current_input = input_ids

    for step in tqdm(range(max_new_tokens), desc='Warm Up Stage'):
        if not torch.any(unfinished):
            break

        with torch.no_grad():
            out = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = out.logits[:, -1, :]  # (batch_size, vocab_size)
            past_key_values = out.past_key_values

        log_probs = F.log_softmax(logits, dim=-1)
        topk_vals, _ = torch.topk(log_probs, k=k_conf, dim=-1)
        Ci_batch = -topk_vals.mean(dim=-1)  # (batch_size,)

        all_confs[step] = torch.where(unfinished, Ci_batch, torch.zeros_like(Ci_batch))

        if temperature == 0.0:
            idx_next = torch.topk(logits, k=1, dim=-1)[1]
        else:
            lp = logits / temperature
            k_sampling = top_k_sampling if top_k_sampling is not None else 0
            if k_sampling > 0 or top_p < 1.0:
                lp = top_k_top_p_filtering(lp.clone(), top_k=k_sampling, top_p=top_p)
            probs = F.softmax(lp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        next_token_ids = idx_next.squeeze(-1)

        just_finished = (next_token_ids == eos) & unfinished
        finished_at_step = torch.where(just_finished,
                                       torch.full_like(finished_at_step, step + 1),
                                       finished_at_step)
        unfinished = unfinished & (next_token_ids != eos)

        current_input = idx_next

    min_gconfs = []
    for i in range(batch_size):
        seq_len = finished_at_step[i].item()
        if seq_len == 0:
            min_gconfs.append(float('inf'))
            continue

        confs = all_confs[:seq_len, i]
        if group_size > 0 and seq_len >= group_size:
            padded = F.pad(confs.unsqueeze(0), (group_size - 1, 0), mode='constant', value=0)
            windows = padded.unfold(1, group_size, 1).squeeze(0)
            indices_i = torch.arange(seq_len, device=DEVICE).unsqueeze(1)  # (seq_len, 1)
            indices_j = torch.arange(group_size, device=DEVICE)  # (group_size,)
            mask = (indices_i + indices_j) >= (group_size - 1)
            valid_counts = mask.sum(dim=1).clamp(min=1).float()
            window_sums = (windows * mask.float()).sum(dim=1)
            group_confs = window_sums / valid_counts
        else:
            cumsum = torch.cumsum(confs, dim=0)
            indices = torch.arange(1, seq_len + 1, device=DEVICE, dtype=torch.float)
            group_confs = cumsum / indices

        min_gconfs.append(group_confs.min().item())

    return min_gconfs


# ------------------ offline warmup to estimate threshold s ------------------
min_gconfs = []
if manual_threshold is not None:
    s = float(manual_threshold)
    print(f"[warmup skipped] using manual threshold s={s:.6f}")
elif N_warmup > 0:
    print(f"[warmup] running {N_warmup} warmup traces to estimate threshold (eta={eta}) ...")
    min_gconfs = generate_traces_batched(model, base_input_ids, max_new_tokens, N_warmup)
    s = float(np.percentile(np.array(min_gconfs), 100.0 - eta))
    print(f"[warmup] done. sample min_gconfs count={len(min_gconfs)}. threshold s={s:.6f}")

# ------------------ main generation: generate num_traces traces with early stopping ------------------
generated_traces = []
print(f"[generate] generating {num_traces} trace(s) with early-stop threshold s={s:.6f} ...")
for t in range(num_traces):
    ids_t, confs_t, min_g_t, CompleteTrajectory = generate_trace_with_conf(model, base_input_ids, max_new_tokens, stop_threshold=s)
    if CompleteTrajectory:
        decoded = tokenizer.decode(ids_t[:, base_input_ids.shape[1]:][0], skip_special_tokens=True)
        generated_traces.append({
            'ids': ids_t,
            'confs': confs_t,
            'min_group_conf': min_g_t,
            'decoded': decoded
        })
        print(f"[trace {t + 1}] tokens_generated={ids_t.shape[1] - base_input_ids.shape[1]}, min_group_conf={min_g_t:.6f}")
        # print(f"[trace {t + 1}] decoded:\n{decoded}\n{'-' * 60}")
    else:
        print(f"[trace {t + 1} early stop!] tokens_generated={ids_t.shape[1] - base_input_ids.shape[1]}, min_group_conf={min_g_t:.6f}")

if generated_traces:
    best_trace = max(generated_traces, key=lambda x: x['min_group_conf'])
    print("\n\n------ Best Decoded (Max Confidence) ------")
    print(best_trace['decoded'])
    print("-" * 40)

if len(min_gconfs) > 0:
    print("[warmup stats] sample min_gconfs (first 10):", min_gconfs[:10])
