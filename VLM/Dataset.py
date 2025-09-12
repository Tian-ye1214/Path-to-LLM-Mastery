import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL.Image import Resampling
from typing import List, Dict, Any


def find_assistant_tokens(tokenizer, target):
    result = []
    start_index = 0
    end_index = 0
    while start_index <= len(target) - 1:
        if target[start_index] != tokenizer('assistant')['input_ids'][0]:
            start_index += 1
            end_index += 1
        else:
            end_index += 1
            if target[end_index] == tokenizer('<|im_end|>')['input_ids'][0]:
                result.append((start_index + 1, end_index + 1))
                start_index = end_index + 1
    return result


class MyDataset(Dataset):
    def __init__(self, data_paths, tokenizer, processor, config):
        super().__init__()
        datasets_list = [load_dataset(data_paths, cache_dir='/root/autodl-tmp/Dataset/llava-cache')['train']]
        from datasets import concatenate_datasets
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
        pixel_values = None
        if image is not None:
            image = image.resize((384, 384), Resampling.BILINEAR).convert('RGB')
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']
            if '<image>' not in conversations[0]['value']:
                conversations[0]['value'] = '<image>\n' + conversations[0]['value']

            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
        else:
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        input_ids = self.tokenizer(text)['input_ids']
        indexs = find_assistant_tokens(self.tokenizer, input_ids)
        labels = len(input_ids) * [self.tokenizer.pad_token_id]
        for index in indexs:
            labels[index[0]:index[1]] = input_ids[index[0]:index[1]]
        input_ids = input_ids[:-1]
        labels = labels[1:]

        max_length = 1536
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


class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values_list = []
        has_image_features = [feature['pixel_values'] is not None for feature in features]

        if any(has_image_features):
            first_valid_pixel_values = next(f['pixel_values'] for f in features if f['pixel_values'] is not None)
            dummy_pixel_values = torch.zeros_like(first_valid_pixel_values)
            for feature in features:
                pixel_values_list.append(feature['pixel_values'] if feature['pixel_values'] is not None else dummy_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None

        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))

        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': pixel_values
        }
        return result