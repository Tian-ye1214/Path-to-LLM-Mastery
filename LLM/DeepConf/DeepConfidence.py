# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    device_map="auto"
)

q_text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": 'Give me a short introduction to large language model.'}
    ],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

base_input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'].to(DEVICE)
base_attention_mask = torch.ones_like(base_input_ids, device=DEVICE)

max_new_tokens = 10240
temperature = 0.6
top_k_sampling = 20
top_p = 0.95
eos = tokenizer.eos_token_id

N_warmup = 16  # 离线 warmup traces 数量。设为 0 跳过 warmup（用 manual_threshold）
eta = 90  # 保留 top-eta% traces，阈值 s = percentile(100-eta)
group_size = 1024  # 滑窗大小
k_conf = 10  # token confidence 使用 top-k logprobs 的平均（k_conf >=1）
num_traces = 16  # 主生成时要生成的 trace 数
manual_threshold = None


def generate_trace_with_conf(model, base_input_ids, max_new_tokens, stop_threshold=None):
    """
    生成一条 trace；记录每个 token 的 token-confidence Ci = -mean(top_k logprobs)
    计算 sliding group confidence = mean of last `group_size` Ci
    若提供 stop_threshold，当 group_conf < stop_threshold 时提前停止（不将该 step 的 token 加入）
    返回：generated_ids (torch.LongTensor), conf_list (list), min_group_conf (float)
    """
    input_ids = base_input_ids.clone()
    attention_mask = torch.ones_like(input_ids, device=DEVICE)
    conf_list = []
    group_confs = []
    CompleteTrajectory = True

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]  # shape (1, V)
        # token log-probs -> top-k for confidence
        log_probs = F.log_softmax(logits, dim=-1)  # (1, V)
        k = min(k_conf, log_probs.size(-1))
        topk_vals, topk_idx = torch.topk(log_probs, k=k, dim=-1)  # (1, k)
        Ci = - float(topk_vals.mean().item())
        conf_list.append(Ci)

        # compute group conf (sliding window mean of Ci)
        window = conf_list[-group_size:] if group_size > 0 else conf_list
        gc = float(np.mean(window))
        group_confs.append(gc)

        # pick next token following original sampling/greedy logic
        if temperature == 0.0:
            _, idx_next = torch.topk(logits, k=1, dim=-1)  # greedy (1,1)
        else:
            lp = logits / temperature
            k = top_k_sampling if top_k_sampling is not None else 0
            if k > 0 or top_p < 1.0:
                lp = top_k_top_p_filtering(lp, top_k=k, top_p=top_p)

            probs = F.softmax(lp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        next_token_id = int(idx_next[0, 0].item())

        # online early-stop: if threshold set and current group_conf < threshold -> stop (do not append token)
        if (stop_threshold is not None) and (gc < stop_threshold):
            CompleteTrajectory = False
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), device=DEVICE, dtype=attention_mask.dtype)),
            dim=1)

        # if generated eos, stop (keep eos)
        if next_token_id == eos:
            break

    min_gconf = float(np.min(group_confs)) if len(group_confs) > 0 else float('inf')
    return input_ids, conf_list, min_gconf, CompleteTrajectory


def generate_traces_batched(model, base_input_ids, max_new_tokens, num_traces):
    """
    并行生成多条 traces，用于 warmup
    """
    batch_size = num_traces
    input_ids = base_input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids, device=DEVICE)

    conf_lists = [[] for _ in range(batch_size)]
    group_confs_lists = [[] for _ in range(batch_size)]
    min_gconfs = [float('inf')] * batch_size

    unfinished = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)

    for step in range(max_new_tokens):
        if not torch.any(unfinished):
            break

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]

        log_probs = F.log_softmax(logits, dim=-1)
        k = min(k_conf, log_probs.size(-1))
        topk_vals, _ = torch.topk(log_probs, k=k, dim=-1)
        Ci_batch = -topk_vals.mean(dim=-1)

        if temperature == 0.0:
            idx_next = torch.topk(logits, k=1, dim=-1)[1]
        else:
            lp = logits / temperature
            k_sampling = top_k_sampling if top_k_sampling is not None else 0
            if k_sampling > 0 or top_p < 1.0:
                lp = top_k_top_p_filtering(lp, top_k=k_sampling, top_p=top_p)
            probs = F.softmax(lp, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        next_token_ids = idx_next.squeeze(-1)

        for i in range(batch_size):
            if unfinished[i]:
                conf_lists[i].append(Ci_batch[i].item())
                window = conf_lists[i][-group_size:] if group_size > 0 else conf_lists[i]
                gc = float(np.mean(window))
                group_confs_lists[i].append(gc)

                if next_token_ids[i] == eos:
                    unfinished[i] = False
                    if len(group_confs_lists[i]) > 0:
                        min_gconfs[i] = float(np.min(group_confs_lists[i]))

        input_ids = torch.cat((input_ids, idx_next), dim=1)
        attention_mask = torch.cat(
            (attention_mask, torch.ones((batch_size, 1), device=DEVICE, dtype=attention_mask.dtype)),
            dim=1
        )

    for i in range(batch_size):
        if unfinished[i]:
            if len(group_confs_lists[i]) > 0:
                min_gconfs[i] = float(np.min(group_confs_lists[i]))

    return min_gconfs


# ------------------ offline warmup to estimate threshold s ------------------
min_gconfs = []
if N_warmup and N_warmup > 0:
    print(f"[warmup] running {N_warmup} warmup traces to estimate threshold (eta={eta}) ...")
    min_gconfs = generate_traces_batched(model, base_input_ids, max_new_tokens, N_warmup)
    s = float(np.percentile(np.array(min_gconfs), 100.0 - eta))
    print(f"[warmup] done. sample min_gconfs count={len(min_gconfs)}. threshold s={s:.6f}")
elif manual_threshold is not None:
    s = float(manual_threshold)
    print(f"[warmup skipped] using manual threshold s={s:.6f}")

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
