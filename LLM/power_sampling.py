# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import math


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


class PowerSampler:
    def __init__(
        self,
        model_name: str,
        device: str = None,
        alpha: float = 4.0,           # 幂分布指数
        block_size: int = 192,         # 块大小 B
        n_mcmc_steps: int = 10,        # MCMC步数
        max_tokens: int = 4096,        # 最大生成长度
        proposal_temp: float = None,   # 提案分布温度, 默认 1/alpha
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )
        self.model.eval()

        self.alpha = alpha
        self.block_size = block_size
        self.n_mcmc_steps = n_mcmc_steps
        self.max_tokens = max_tokens
        self.proposal_temp = proposal_temp if proposal_temp is not None else (1.0 / alpha)
        
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def compute_log_likelihood(
        self, 
        input_ids: torch.Tensor,
        start_idx: int = 0
    ) -> torch.Tensor:
        """
        计算序列的对数似然 log p(x_start_idx:T | x_0:start_idx-1)
        
        Args:
            input_ids: 完整序列 [1, seq_len]
            start_idx: 开始计算的位置
        
        Returns:
            log_likelihood: 对数似然值
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        log_probs = F.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
        total_log_prob = 0.0
        seq_len = input_ids.shape[1]
        
        for t in range(max(1, start_idx), seq_len):
            token_id = input_ids[0, t].item()
            total_log_prob += log_probs[0, t-1, token_id].item()
            
        return total_log_prob
    
    def sample_with_proposal(
        self,
        prefix_ids: torch.Tensor,
        num_tokens: int,
        allow_early_stop: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        使用提案分布(低温采样)生成continuation
        
        Args:
            prefix_ids: 前缀token ids [1, prefix_len]
            num_tokens: 要生成的token数量
            allow_early_stop: 是否允许遇到EOS提前结束（MCMC步骤中应为False）
            
        Returns:
            full_sequence: 完整序列(前缀+生成)
            proposal_log_prob: 提案分布下的对数概率
        """
        current_input = prefix_ids.clone()
        generated_tokens = []
        proposal_log_prob = 0.0
        past_key_values = None
        
        for step in range(num_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input if past_key_values is None else current_input[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[:, -1, :].clone()
                past_key_values = outputs.past_key_values

                logits = top_k_top_p_filtering(logits / self.proposal_temp, top_k=20, top_p=0.95)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token[0, 0].item()

                proposal_log_prob += log_probs[0, next_token_id].item()
                
                generated_tokens.append(next_token_id)
                current_input = torch.cat([current_input, next_token], dim=-1)

                if allow_early_stop and next_token_id == self.eos_token_id:
                    break
        
        return current_input, proposal_log_prob
    
    def metropolis_hastings_step(
        self,
        current_seq: torch.Tensor,
        original_prompt_len: int,
        target_seq_len: int,
        current_log_prob: float
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        执行一步Metropolis-Hastings采样
        
        根据论文Algorithm 1:
        1. 随机选择一个重采样位置 m ∈ {1, ..., (k+1)B}（相对于生成序列）
        2. 从位置m开始用提案分布重新生成
        3. 根据接受率决定是否接受新序列
        
        Args:
            current_seq: 当前序列 [1, seq_len]
            original_prompt_len: 原始prompt长度（不可重采样的部分）
            target_seq_len: 目标序列长度（确保重采样后长度一致）
            current_log_prob: 当前序列的对数似然
            
        Returns:
            new_seq: 新序列(可能与current_seq相同)
            new_log_prob: 新序列的对数似然
            accepted: 是否接受了新序列
        """
        seq_len = current_seq.shape[1]
        resampling_len = seq_len - original_prompt_len
        
        if resampling_len <= 1:
            return current_seq, current_log_prob, False
        
        # 1. 随机选择重采样位置 m ∈ [original_prompt_len+1, seq_len)
        m = torch.randint(original_prompt_len + 1, seq_len, (1,)).item()
        
        # 2. 保留前缀 x_0:m-1, 从位置m开始重新采样
        prefix = current_seq[:, :m]
        tokens_to_generate = target_seq_len - m
        
        # 使用提案分布生成新的continuation
        proposal_seq, proposal_forward_log_prob = self.sample_with_proposal(
            prefix, tokens_to_generate, allow_early_stop=False
        )
        
        # 3. 计算接受率
        # A(x', x) = min(1, π(x')/π(x) * q(x|x')/q(x'|x))
        # 其中 π(x) ∝ p(x)^α
        
        # 计算提案序列在基础模型下的对数似然
        proposal_base_log_prob = self.compute_log_likelihood(proposal_seq, start_idx=0)
        
        # 计算反向提案概率 q(x|x') - 即从proposal_seq重新采样得到current_seq的概率
        # 由于对称性,这等于从相同前缀生成current_seq后半部分的概率
        # 我们需要计算 current_seq[m:] 在提案分布下的对数概率
        current_suffix = current_seq[:, m:]
        reverse_log_prob = self._compute_proposal_log_prob(prefix, current_suffix)
        
        # 幂分布的对数似然
        proposal_power_log_prob = self.alpha * proposal_base_log_prob
        current_power_log_prob = self.alpha * current_log_prob
        
        # 对数接受率
        log_accept_ratio = proposal_power_log_prob - current_power_log_prob + reverse_log_prob - proposal_forward_log_prob

        
        # 4. 以概率min(1, exp(log_accept_ratio))接受
        log_u = math.log(torch.rand(1).item() + 1e-10)
        
        if log_u < log_accept_ratio:
            # 接受新序列
            return proposal_seq, proposal_base_log_prob, True
        else:
            # 保持当前序列
            return current_seq, current_log_prob, False
    
    def _compute_proposal_log_prob(
        self,
        prefix: torch.Tensor,
        suffix: torch.Tensor
    ) -> float:
        """
        计算在给定前缀下,用提案分布生成suffix的对数概率
        """
        full_seq = torch.cat([prefix, suffix], dim=-1)
        
        with torch.no_grad():
            outputs = self.model(input_ids=full_seq, use_cache=False)
            logits = outputs.logits / self.proposal_temp
            log_probs = F.log_softmax(logits, dim=-1)
        
        prefix_len = prefix.shape[1]
        total_log_prob = 0.0
        
        for t in range(prefix_len, full_seq.shape[1]):
            token_id = full_seq[0, t].item()
            total_log_prob += log_probs[0, t-1, token_id].item()
            
        return total_log_prob
    
    def power_sample(
        self,
        prompt: str,
    ) -> str:
        """
        主采样函数: 实现Algorithm 1 - Power Sampling for Autoregressive Models
        
        核心流程:
        1. 将生成过程分成多个block
        2. 对每个block,先用提案分布初始化,然后执行MCMC采样
        3. 逐步从中间分布 π_k 过渡到目标分布 p^α
        
        Args:
            prompt: 输入提示文本
        Returns:
            生成的文本
        """
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        prefix_len = input_ids.shape[1]
        num_blocks = math.ceil(self.max_tokens / self.block_size)
        current_seq = input_ids

        print(f"开始Power Sampling: α={self.alpha}, B={self.block_size}, N_MCMC={self.n_mcmc_steps}")
        print(f"预计生成 {num_blocks} 个block")
        
        for k in range(num_blocks):
            target_len = prefix_len + (k + 1) * self.block_size
            target_len = min(target_len, prefix_len + self.max_tokens)
            
            current_len = current_seq.shape[1]
            tokens_to_add = target_len - current_len
            
            if tokens_to_add <= 0:
                break

            current_seq, _ = self.sample_with_proposal(current_seq, tokens_to_add)

            if self.eos_token_id in current_seq[0, current_len:].tolist():
                print(f"Block {k}: 遇到EOS token,提前结束")
                break

            current_log_prob = self.compute_log_likelihood(current_seq, start_idx=0)

            print(f"Block {k}: 初始化完成, 序列长度={current_seq.shape[1]}, log_prob={current_log_prob:.2f}")

            accept_count = 0
            target_seq_len = current_seq.shape[1]  # 记录目标长度，确保MCMC过程中长度一致
            
            for n in range(self.n_mcmc_steps):
                current_seq, current_log_prob, accepted = self.metropolis_hastings_step(
                    current_seq, 
                    prefix_len,           # 原始prompt长度，m可以从任意生成位置开始
                    target_seq_len,       # 目标序列长度，确保重采样后长度一致
                    current_log_prob
                )
                if accepted:
                    accept_count += 1

                if self.eos_token_id in current_seq[0, prefix_len:].tolist():
                    break

            accept_rate = accept_count / self.n_mcmc_steps * 100
            print(f"Block {k}: MCMC完成, 接受率={accept_rate:.1f}%, 最终log_prob={current_log_prob:.2f}")
            
            # 检查是否达到最大长度或遇到EOS
            if current_seq.shape[1] >= prefix_len + self.max_tokens:
                break
            if self.eos_token_id in current_seq[0, prefix_len:].tolist():
                break
        
        # 解码输出
        generated_ids = current_seq[0, prefix_len:].tolist()
        
        # 截断到EOS
        if self.eos_token_id in generated_ids:
            eos_idx = generated_ids.index(self.eos_token_id)
            generated_ids = generated_ids[:eos_idx]
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text



if __name__ == "__main__":
    model_name = "G:/代码/ModelWeight/Qwen3-0.6B"

    sampler = PowerSampler(
        model_name=model_name,
        alpha=4.0,           # 幂分布指数
        block_size=64,       # 块大小(小模型可以用更小的block)
        n_mcmc_steps=5,      # MCMC步数
        max_tokens=4096,      # 最大生成长度
        proposal_temp=0.6
    )
    prompt = sampler.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \le \\theta < 2 \pi.$"}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    result = sampler.power_sample(prompt)
    print("\n生成结果:")
    print(result)

