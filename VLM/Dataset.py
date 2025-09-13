import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL.Image import Resampling
from PIL import Image
from transformers import DataCollatorForSeq2Seq
from datasets import concatenate_datasets


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    assistant_id = tokenizer('assistant')['input_ids'][0]
    im_end_id = tokenizer('<|im_end|>')['input_ids'][0]

    while start_index < len(target):
        if target[start_index] != assistant_id:
            start_index += 1
            continue

        end_index = start_index + 1
        found_end = False

        while end_index < len(target):
            if target[end_index] == im_end_id:
                result.append((start_index + 1, end_index + 1))
                found_end = True
                break
            end_index += 1

        if not found_end:
            result.append((start_index + 1, len(target)))
        start_index = end_index + 1

    return result


class MyDataset(Dataset):
    def __init__(self, data_paths, tokenizer, processor, config):
        super().__init__()
        datasets_list = [load_dataset(data_paths, cache_dir='/root/autodl-tmp/Dataset/llava-cache')['train']]
        self.datas = concatenate_datasets(datasets_list).shuffle(seed=42)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        image = sample['image']
        conversations = sample['conversations']
        messages = [{"role": "system", "content": 'You are a helpful assistant.'}]
        if image is not None:
            image = image.resize((384, 384), Resampling.BILINEAR).convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='white')

        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']
        if '<image>' not in conversations[0]['value']:
            conversations[0]['value'] = '<image>\n' + conversations[0]['value']

        for conversation in conversations:
            if conversation['from'] == 'human':
                messages.append({"role": "user", "content": conversation['value']})
            else:
                messages.append({"role": "assistant", "content": conversation['value']})
        text = (self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                .replace('<image>', '<|vision_start|>' + '<|image_pad|>' * self.config.image_pad_num + '<|vision_end|>')
                )

        input_ids = self.tokenizer(text)['input_ids']
        indexs = find_assistant_tokens(self.tokenizer, input_ids)
        labels = len(input_ids) * [self.tokenizer.pad_token_id]
        for index in indexs:
            labels[index[0]:index[1]] = input_ids[index[0]:index[1]]

        max_length = 4096
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        input_ids = input_ids[:-1]
        labels = labels[1:]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        }


class MyDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.cat([feature.pop("pixel_values") for feature in features], dim=0)
        batch = super().__call__(features, return_tensors)
        batch["pixel_values"] = pixel_values
        return batch
