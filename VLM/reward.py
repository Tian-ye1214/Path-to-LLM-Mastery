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

        if not think_match or not answer_match:
            rewards.append(0.0)
            continue
        
        think_content = think_match.group(1).strip()
        answer_content = answer_match.group(1).strip()
        
        if not answer_content:
            rewards.append(0.0)
            continue

        if not think_content:
            rewards.append(0.3)
        elif len(think_content) < 50:
            rewards.append(0.7)
        else:
            rewards.append(1.0)
    
    return rewards

def language_consistency_reward(completions, **kwargs):
    rewards = []
    
    for content in completions:
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        total_chars = chinese_chars + english_chars

        if total_chars == 0:
            rewards.append(0.0)
            continue

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        # 设置阈值：如果某种语言占比超过95%，认为是单一语言
        threshold = 0.95
        
        if chinese_ratio >= threshold or english_ratio >= threshold:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def self_verification_reward(completions, **kwargs):
    """Reward function that encourages self-verification in reasoning.
    
    Reward structure:
    - 0.0: No verification behavior
    - 0.5: Mentions verification but doesn't execute
    - 1.0: Contains actual verification steps (substitution, reverse reasoning, etc.)
    """
    rewards = []
    
    # 验证相关的关键词
    verification_keywords_cn = [
        '验证', '检验', '检查', '代入', '反推', '核对', '确认',
        '验算', '回代', '带入', '证明', '推导'
    ]
    verification_keywords_en = [
        'verify', 'check', 'validation', 'substitute', 'confirm',
        'proof', 'test', 'prove', 'double-check', 'ensure'
    ]
    
    # 实际验证行为的模式（包含具体的验证步骤）
    actual_verification_patterns = [
        r'代入.*[=＝]',  # 代入并显示等式
        r'验证[:：].*[=＝]',  # 验证：后面有等式
        r'检查[:：].*[=＝]',  # 检查：后面有等式
        r'substitute.*[=]',  # 英文代入
        r'verify.*[=]',  # 英文验证
        r'反推.*[=＝]',  # 反向推理
        r'(满足|符合|成立)',  # 验证结果的表达
    ]
    
    for content in completions:
        score = 0.0
        content_lower = content.lower()
        
        # 检查是否包含验证关键词
        has_verification_keyword = False
        for keyword in verification_keywords_cn + verification_keywords_en:
            if keyword in content_lower:
                has_verification_keyword = True
                break
        
        if has_verification_keyword:
            # 至少提到了验证，给0.5分
            score = 0.5
            for pattern in actual_verification_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    score = 1.0
                    break
        
        rewards.append(score)
    
    return rewards

def conciseness_reward(completions, **kwargs):
    """Reward function that encourages concise and non-redundant responses.
    
    Penalizes:
    - Excessive repetition
    - Redundancy between <think> and <answer>
    - Unreasonable length
    
    Reward structure:
    - 1.0: Concise and clear
    - 0.5-0.8: Minor issues
    - 0.0-0.4: Severe redundancy or length issues
    """
    rewards = []
    
    for content in completions:
        score = 1.0
        
        # 提取think和answer部分
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', content, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        
        if not think_match or not answer_match:
            rewards.append(0.5)
            continue
        
        think_content = think_match.group(1).strip()
        answer_content = answer_match.group(1).strip()
        
        # 1. 检查重复句子（连续相同的句子）
        sentences = re.split(r'[。！？\.\!\?]', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) > 1:
            repeated_count = 0
            for i in range(len(sentences) - 1):
                if sentences[i] == sentences[i + 1]:
                    repeated_count += 1
            
            if repeated_count > 0:
                score -= 0.3 * min(repeated_count, 2)  # 最多扣0.6分
        
        # 2. 检查think和answer之间的重复度
        if think_content and answer_content:
            # 计算answer内容在think中的重复比例
            answer_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', answer_content))
            think_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', think_content))
            
            if len(answer_words) > 0:
                overlap_ratio = len(answer_words & think_words) / len(answer_words)
                # 如果answer几乎完全在think中出现过（>90%），说明过于重复
                if overlap_ratio > 0.9 and len(answer_content) > 20:
                    score -= 0.3
        
        # 3. 检查总长度是否合理
        total_length = len(content)
        if total_length > 3000:  # 过长
            score -= 0.2
        elif total_length > 5000:  # 非常长
            score -= 0.4
        
        # 4. 检查是否有大段重复内容（相同的长文本块）
        # 使用滑动窗口检测重复片段
        chunk_size = 50
        if len(content) > chunk_size * 2:
            chunks = []
            for i in range(0, len(content) - chunk_size, 10):
                chunks.append(content[i:i + chunk_size])
            
            unique_chunks = len(set(chunks))
            total_chunks = len(chunks)
            
            if total_chunks > 0:
                uniqueness_ratio = unique_chunks / total_chunks
                if uniqueness_ratio < 0.7:  # 重复度很高
                    score -= 0.3
        
        rewards.append(max(0.0, score))
    
    return rewards

def confidence_reward(completions, **kwargs):
    """Reward function that encourages appropriate expression of uncertainty.
    
    Reward structure:
    - 1.0: Appropriate confidence expression (certain when should be, uncertain when appropriate)
    - 0.7: Overly confident or overly uncertain
    - 0.5: No confidence expression when needed
    """
    rewards = []
    
    # 不确定性表达的关键词
    uncertainty_keywords = [
        # 中文
        '可能', '大概', '大约', '约', '估计', '推测', '猜测',
        '也许', '或许', '应该', '似乎', '看起来', '假设',
        '不确定', '难以确定', '需要更多信息',
        # 英文
        'may', 'might', 'probably', 'possibly', 'perhaps',
        'approximately', 'roughly', 'about', 'around',
        'assume', 'uncertain', 'unclear', 'not sure'
    ]
    
    # 高置信度表达
    high_confidence_keywords = [
        # 中文
        '一定', '必然', '肯定', '确定', '毫无疑问', '显然',
        '明确', '确实', '绝对', '无疑',
        # 英文
        'definitely', 'certainly', 'absolutely', 'surely',
        'clearly', 'obviously', 'undoubtedly', 'must be'
    ]
    
    # 条件性表达
    conditional_keywords = [
        # 中文
        '如果', '假如', '若', '倘若', '当', '在.*情况下',
        '前提是', '基于', '根据',
        # 英文
        'if', 'assuming', 'given', 'provided', 'when',
        'under the condition', 'based on'
    ]
    
    for content in completions:
        score = 1.0  # 默认满分
        content_lower = content.lower()
        
        # 计算各类关键词出现次数
        uncertainty_count = sum(1 for kw in uncertainty_keywords if kw in content_lower)
        confidence_count = sum(1 for kw in high_confidence_keywords if kw in content_lower)
        conditional_count = sum(1 for kw in conditional_keywords if kw in content_lower)
        
        # 分析内容长度
        content_length = len(content)
        
        # 评估策略：
        # 1. 有条件性表达是好的（说明考虑了前提）
        if conditional_count > 0:
            score = 1.0
        
        # 2. 如果回答很短(<100字符)但使用很多不确定词，可能过于谨慎
        elif content_length < 100 and uncertainty_count > 3:
            score = 0.7
        
        # 3. 如果回答很长但完全没有任何置信度表达，可能问题
        elif content_length > 500 and uncertainty_count == 0 and conditional_count == 0:
            # 检查是否全是高置信度表达（可能过于自信）
            if confidence_count > 3:
                score = 0.7
            else:
                score = 0.9  # 轻微扣分
        
        # 4. 平衡使用不确定和确定表达
        elif uncertainty_count > 0 and confidence_count > 0:
            score = 1.0  # 有nuance的表达
        
        rewards.append(score)
    
    return rewards

def error_penalty_reward(completions, **kwargs):
    """Reward function that penalizes common errors.
    
    Penalizes:
    - Contradictions between <think> and <answer>
    - Incomplete markers (TODO, ..., 待补充)
    - Nonsense or garbled text
    - Answer that doesn't match question
    
    Reward structure:
    - 1.0: No errors detected
    - 0.0-0.8: Errors detected (penalties applied)
    """
    rewards = []
    
    # 未完成标记
    incomplete_markers = [
        'TODO', 'todo', 'FIXME', 'XXX', 
        '待补充', '待完成', '未完成', '省略',
        '...', '。。。', '···',
    ]
    
    # 乱码模式
    garbled_patterns = [
        r'[^\u4e00-\u9fff\u0000-\u007F\s\d\W]{10,}',  # 连续10个以上奇怪字符
        r'(\w)\1{10,}',  # 同一字符重复10次以上
    ]
    
    for content in completions:
        score = 1.0
        
        # 提取think和answer部分
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', content, re.DOTALL)
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
        
        if not think_match or not answer_match:
            rewards.append(0.5)
            continue
        
        think_content = think_match.group(1).strip()
        answer_content = answer_match.group(1).strip()
        
        # 1. 检查未完成标记
        for marker in incomplete_markers:
            if marker in content:
                score -= 0.4
                break
        
        # 2. 检查乱码
        for pattern in garbled_patterns:
            if re.search(pattern, content):
                score -= 0.5
                break
        
        # 3. 检查think和answer之间的矛盾
        # 提取think中的数字和answer中的数字
        think_numbers = re.findall(r'\d+\.?\d*', think_content)
        answer_numbers = re.findall(r'\d+\.?\d*', answer_content)
        
        # 如果think中得出的最后几个数字和answer中的数字完全不同，可能矛盾
        if len(think_numbers) > 0 and len(answer_numbers) > 0:
            # 取think中最后3个数字
            last_think_numbers = set(think_numbers[-3:])
            answer_number_set = set(answer_numbers)
            
            # 如果完全没有交集，可能存在矛盾
            if len(last_think_numbers & answer_number_set) == 0:
                # 进一步检查：是否think中明确说了结果
                if re.search(r'(答案|结果|得出|因此|所以|answer|result|therefore).*\d', think_content):
                    score -= 0.3
        
        # 4. 检查answer是否为空或过于简短
        if len(answer_content) < 1:
            score -= 0.5
        elif len(answer_content) < 3 and not answer_content.isdigit():
            # 如果answer很短但不是数字答案，可能有问题
            score -= 0.2
        
        # 5. 检查是否有明显的自相矛盾词汇
        contradiction_patterns = [
            (r'正确', r'错误'),
            (r'是', r'不是'),
            (r'可以', r'不可以'),
            (r'True', r'False'),
            (r'yes', r'no'),
        ]
        
        for positive, negative in contradiction_patterns:
            if re.search(positive, think_content, re.IGNORECASE) and \
               re.search(negative, answer_content, re.IGNORECASE):
                # 可能存在矛盾，但也可能是正常的否定句，所以轻微惩罚
                score -= 0.1
                break
        
        rewards.append(max(0.0, score))
    
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
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards
