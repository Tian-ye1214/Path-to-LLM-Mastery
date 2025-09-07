from pathlib import Path
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.normalizers import NFKC, Sequence
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(args) -> None:
    original_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None
    ).to("cuda")
    print('分词器类型', type(original_tokenizer).__name__)
    print('原始模型架构', model)

    special_tokens = list(original_tokenizer.all_special_tokens)
    unk_token = original_tokenizer.unk_token

    new_bpe_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    new_bpe_tokenizer.normalizer = Sequence([NFKC()])
    new_bpe_tokenizer.pre_tokenizer = ByteLevel() if args.use_bytes_level else Whitespace()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,         # 词汇表大小，指定训练后词汇表的最大词条数量
        min_frequency=args.min_frequency,   # 最小频率，只有出现次数大于等于此值的词条才会被加入词汇表
        special_tokens=special_tokens,      # 特殊标记列表，这些标记会被强制添加到词汇表中
    )

    corpus_path = Path(args.data_path)
    new_bpe_tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    new_vocab = new_bpe_tokenizer.get_vocab()
    original_vocab = original_tokenizer.get_vocab()
    added_tokens = [tok for tok in new_vocab.keys() if tok not in original_vocab]

    num_added_toks = 0
    if added_tokens:
        num_added_toks = original_tokenizer.add_tokens(added_tokens)

    new_vocab_size = len(original_tokenizer)
    print(f"原始词表大小: {len(original_vocab)}")
    print(f"新词表大小: {new_vocab_size}")
    print(f"新增词条数: {num_added_toks}")

    model.resize_token_embeddings(new_vocab_size, mean_resizing=True)
    print('修改后模型架构', model)

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    original_tokenizer.save_pretrained(args.output_path)
    model.save_pretrained(args.output_path)
    print(f"已保存模型至: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='参数设置')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='', help='模型权重路径')
    parser.add_argument('--data_path', type=str, default='./merged_output.txt', help='训练数据路径')
    parser.add_argument('--output_path', type=str, default='./updated_qwen_tokenizer', help='输出数据路径')
    parser.add_argument('--use_bytes_level', type=bool, default=True, help='是否使用BBPE')
    parser.add_argument('--vocab_size', type=int, default=4096, help='最大拓展词表大小')
    parser.add_argument('--min_frequency', type=int, default=2, help='最小积累')

    args = parser.parse_args()
    main(args)
