from accelerate import Accelerator
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List, Dict, Any


class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"

    def __init__(self, llm_model_path='/root/autodl-tmp/ModelCheckpoint/Qwen3',
                 vision_model_path='/root/autodl-tmp/ModelCheckpoint/Dinov3',
                 freeze_vision_model=True,
                 freeze_llm_model=False,
                 image_pad_num=201,
                 **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model = freeze_llm_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)


class VLM(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path, low_cpu_mem_usage=True, attn_implementation="sdpa")
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path, low_cpu_mem_usage=True, attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path, use_fast=True)
        self.adaper = nn.Sequential(
            nn.RMSNorm(1024),
            nn.Linear(1024, self.llm_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        )

        if self.config.freeze_vision_model:
            print("*" * 20, "Freezing Vision Model", "*" * 20)
            for param in self.vision_model.parameters():
                param.requires_grad = False
        if self.config.freeze_llm_model:
            print("*" * 20, "Freezing LLM Model", "*" * 20)
            for param in self.llm_model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)

        image_embeds = self.vision_model(pixel_values).last_hidden_state

        image_features = self.adaper(image_embeds)

        text_embeds = text_embeds.to(image_features.dtype)

        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])

        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)

        return inputs_embeds


class MyDataset(Dataset):
    def __init__(self, data_paths, tokenizer, processor, config):
        super().__init__()
        datasets_list = [load_dataset(data_paths, 'en')['train'], load_dataset(data_paths, 'zh')['train']]
        from datasets import concatenate_datasets
        self.datas = concatenate_datasets(datasets_list).shuffle(seed=42)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            conversations = sample['messages']
            q_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": 'You are a helpful assistant.'},
                    {"role": "user", "content": conversations[0]['content']}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            a_text = conversations[1]['content'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

            image = sample['images'][0].convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']
        except:
            default_image = Image.new('RGB', (384, 384), color='white')
            pixel_values = self.processor(images=default_image, return_tensors="pt")['pixel_values']
            q_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": 'You are a helpful assistant.'},
                    {"role": "user", "content": "图片内容是什么？<image>"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(
                feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}


if __name__ == '__main__':
    accelerator = Accelerator()
    config = VLMConfig()
    model = VLM(config).half()

    if accelerator.is_main_process:
        print(model)
        print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_path = '/root/autodl-tmp/Dataset/llava'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, use_fast=True)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_strategy='epoch',
        bf16=True,
        gradient_accumulation_steps=16,
        logging_steps=1,
        dataloader_num_workers=50,
        use_liger_kernel=True,
        warmup_ratio=0.1,
        deepspeed='deepspeed_config.json',
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
