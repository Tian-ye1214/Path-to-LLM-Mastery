1. **计算 instruction→output 类型对（单对或数据集）的[困惑度（perplexity）](https://huggingface.co/docs/transformers/perplexity)**，使得只对 `output` 部分求交叉熵（instruction 部分被屏蔽）。
2. **提取模型权重与注意力图（attention maps）** 用作需求可视化
