# QwenoV3

一个基于DINOv3-ViT-L和Qwen3-0.6B的小型视觉语言模型，参数量总计为1B

## 项目概述

QwenoV3结合了DINOv3视觉编码器和Qwen语言模型，实现了多模态理解能力。


## 模型架构

- **视觉编码器**: facebook/dinov3-vitl16-pretrain-lvd1689m
- **语言模型**: Qwen/Qwen3-0.6B
- **融合层**: 两层线性变换

## 数据集

使用[LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data)数据集进行单轮问答预训练、多轮问答SFT

模型权重下载地址：

[Pretrain](https://pan.baidu.com/s/1A2QkAZf2avs-mtV2gD_7YQ?pwd=chif)

[SFT](https://pan.baidu.com/s/1irR0XOWI7_I_6jNcVSRsDw?pwd=3auy)

训练详情：[SwanLab](https://swanlab.cn/@tian_ye/MySmallVLM?utm_source=website_qr&utm_medium=qr_scan)








