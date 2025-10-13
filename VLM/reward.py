from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional
import re
import numpy as np


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
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
        except:
            gold_parsed = []

        if len(gold_parsed) != 0:
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = 0.0
        else:
            reward = 0.0
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
            - cosine_max_len: 最大长度阈值
    """
    import math

    # 获取参数
    min_len_value_wrong = -0.5
    max_len_value_wrong = 0.0
    min_len_value_correct = 1.0
    max_len_value_correct = 0.5
    max_len = 4096
    acc_rewards = accuracy_reward(completions, solution, **kwargs)

    def cosfn(t, T, min_value, max_value):
        """余弦函数计算奖励值"""
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    rewards = []
    for ids, acc_reward in zip(completions, acc_rewards):
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


def soft_overlong_reward(completions, **kwargs):
    soft_max_length = 4096
    soft_cache_length = 2048
    expected_len = soft_max_length - soft_cache_length

    rewards = []
    for completion in completions:
        completion_length = len(completion)
        exceed_len = completion_length - expected_len
        ratio = min(exceed_len / soft_cache_length, 1.0)
        reward = -0.5 * (1 - np.cos(np.pi * ratio))
        rewards.append(reward)

    return rewards

