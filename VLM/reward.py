from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional
import re


def format_reward(completions, **kwargs):
    rewards = []
    for content in completions:
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        answer_pattern = r"<answer>\s*(.*?)\s*</answer>"

        think_match = re.search(think_pattern, content, re.DOTALL)
        answer_match = re.search(answer_pattern, content, re.DOTALL)

        reward = 0.0

        if think_match:
            reward += 0.1
            think_content = think_match.group(1).strip()
            if think_content:
                if len(think_content) < 50:
                    reward += 0.3
                else:
                    reward += 0.7

        if answer_match:
            answer_content = answer_match.group(1).strip()
            if answer_content:
                reward += 1.0
            else:
                reward += 0.3

        rewards.append(reward)

    return rewards


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.
    """
    rewards = []

    for completion, sol in zip(completions, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                is_correct = verify(gold_parsed, answer_parsed)
                reward = 1.0 if is_correct else -1.0
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            is_match = completion.strip().lower() == sol.strip().lower()
            reward = 1.0 if is_match else -1.0

        rewards.append(reward)

    return rewards


def cosine_reward(completions, solution, **kwargs):
    """基于余弦函数的奖励函数，根据答案正确性和生成长度动态调整奖励值
    参考论文：https://arxiv.org/abs/2502.03373
    
    参数：
        completions: 模型生成的完成文本列表
        solution: 标准答案列表
        kwargs: 其他参数，包括：
            - response_token_ids: 生成的token id列表
            - cosine_min_len_value_wrong: 错误答案最小长度时的奖励值（默认-0.5）
            - cosine_max_len_value_wrong: 错误答案最大长度时的奖励值（默认0.0）
            - cosine_min_len_value_correct: 正确答案最小长度时的奖励值（默认1.0）
            - cosine_max_len_value_correct: 正确答案最大长度时的奖励值（默认0.5）
            - cosine_max_len: 最大长度阈值（默认1000）
    """
    import math
    
    # 获取参数
    min_len_value_wrong = kwargs.get('cosine_min_len_value_wrong', -0.5)
    max_len_value_wrong = kwargs.get('cosine_max_len_value_wrong', 0.0)
    min_len_value_correct = kwargs.get('cosine_min_len_value_correct', 1.0)
    max_len_value_correct = kwargs.get('cosine_max_len_value_correct', 0.5)
    max_len = kwargs.get('cosine_max_len', 4096)
    response_token_ids = kwargs.get('response_token_ids', None)
    
    # 首先获取准确性奖励
    acc_rewards = accuracy_reward(completions, solution, **kwargs)
    
    # 如果没有提供token_ids，则只返回准确性奖励
    if response_token_ids is None:
        return acc_rewards
    
    def cosfn(t, T, min_value, max_value):
        """余弦函数计算奖励值"""
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2
    
    rewards = []
    for ids, acc_reward in zip(response_token_ids, acc_rewards):
        if acc_reward is None:
            rewards.append(None)
            continue
            
        is_correct = acc_reward > 0.0
        if is_correct:
            min_value = max_len_value_correct
            max_value = min_len_value_correct
        else:
            min_value = max_len_value_wrong
            max_value = min_len_value_wrong
        
        gen_len = len(ids)
        reward = cosfn(gen_len, max_len, min_value, max_value)
        rewards.append(reward)
    
    return rewards


def repetition_penalty_reward(completions, **kwargs):
    """重复惩罚奖励函数，通过检测n-gram重复来惩罚重复性文本
    参考论文：https://arxiv.org/abs/2502.03373
    
    参数：
        completions: 模型生成的完成文本列表
        kwargs: 其他参数，包括：
            - repetition_n_grams: n-gram大小（默认3）
            - repetition_max_penalty: 最大惩罚值（默认-1.0）
    """
    ngram_size = kwargs.get('repetition_n_grams', 3)
    max_penalty = kwargs.get('repetition_max_penalty', -1.0)
    
    def zipngram(text: str, ngram_size: int):
        """生成文本的n-gram"""
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    
    rewards = []
    for completion in completions:
        # 空文本不惩罚
        if completion == '':
            rewards.append(0.0)
            continue
        
        # 文本太短无法构成n-gram，不惩罚
        if len(completion.split()) < ngram_size:
            rewards.append(0.0)
            continue
        
        # 统计唯一n-gram的比例
        ngrams = set()
        total = 0
        for ng in zipngram(completion, ngram_size):
            ngrams.add(ng)
            total += 1
        
        # 计算重复程度：1 - (唯一n-gram数量 / 总n-gram数量)
        scaling = 1 - len(ngrams) / total
        # 根据重复程度计算惩罚
        reward = scaling * max_penalty
        rewards.append(reward)
    
    return rewards
