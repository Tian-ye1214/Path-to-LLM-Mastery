import torch
from datasets import load_dataset, get_dataset_config_names, interleave_datasets, Features, Image as HFImage, Value, Sequence, IterableDataset as HFIterableDataset
from torch.utils.data import IterableDataset
from PIL.Image import Resampling
from PIL import Image
from transformers import DataCollatorForSeq2Seq
import base64
import io
import requests
import random


def parse_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.resize((224, 224), Resampling.BILINEAR).convert('RGB')

    if isinstance(image, dict):
        if 'bytes' in image:
            try:
                return Image.open(io.BytesIO(image['bytes'])).resize((224, 224), Resampling.BILINEAR).convert('RGB')
            except Exception as e:
                print(f"字节数据解析失败: {e}")
                return None
        elif 'path' in image:
            try:
                return Image.open(image['path']).resize((224, 224), Resampling.BILINEAR).convert('RGB')
            except Exception as e:
                print(f"路径加载失败: {e}")
                return None

    if isinstance(image, str):
        if image.startswith(('http://', 'https://')):
            try:
                response = requests.get(image, timeout=10)
                return Image.open(io.BytesIO(response.content)).resize((224, 224), Resampling.BILINEAR).convert('RGB')
            except Exception as e:
                print(f"URL加载失败: {e}")
                return None
        try:
            if ',' in image and image.startswith('data:'):
                image = image.split(',', 1)[1]
            image_data = base64.b64decode(image)
            return Image.open(io.BytesIO(image_data)).resize((224, 224), Resampling.BILINEAR).convert('RGB')
        except Exception as e:
            print(f"Base64解析失败: {e}")
            return None

    return None


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


def map_VisionArena_format(example):
    conversations = []
    for turn in example['conversation']:
        if isinstance(turn, list) and len(turn) > 0:
            msg = turn[0]
            role_map = {'user': 'human', 'assistant': 'gpt'}
            conversations.append({
                'from': role_map.get(msg.get('role', ''), 'human'),
                'value': msg.get('content', '')
            })

    return {
        'image': parse_image(example['images']),
        'conversations': conversations,
    }


def map_onevision_format(example):
    return {
        'image': parse_image(example['image']),
        'conversations': example['conversations'],
    }


def map_open_r1_format(example):
    conversations = [
        {'from': 'human', 'value': example['problem']},
        {'from': 'gpt', 'value': example['solution']}
    ]
    return {
        'image': parse_image(example['image']),
        'conversations': conversations,
    }


def map_mmmu_format(example):
    questions = example['question'] + '\n' + example['options']
    conversations = [
        {'from': 'human', 'value': questions},
        {'from': 'gpt', 'value': example['answer']}
    ]
    return {
        'image':  parse_image(example['image_1']),
        'conversations': conversations,
    }


def map_cauldron_format(example):
    conversations = []
    for turn in example['texts']:
        conversations.append({'from': 'human', 'value': turn['user']})
        conversations.append({'from': 'gpt', 'value': turn['assistant']})
    return {
        'image':  parse_image(example['images']),
        'conversations': conversations,
    }


def map_livebench_format(example):
    answer = '<think>' + example['reason'] + '</think>\n' + example['answer']
    conversations = [
        {'from': 'human', 'value': example['question']},
        {'from': 'gpt', 'value': answer}
    ]
    return {
        'image':  parse_image(example['images']),
        'conversations': conversations,
    }


def map_llava_format(example):
    return {
        'image': parse_image(example['image']),
        'conversations': example['conversations'],
    }


class MyDataset(IterableDataset):
    def __init__(self, data_paths, tokenizer, processor, config):
        super().__init__()

        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        datasets_list = []
        cache_dir = '/root/autodl-tmp/Dataset/dataset-cache'
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        features = Features({
            'image': HFImage(),
            'conversations': Sequence({
                'from': Value('string'),
                'value': Value('string')
            })
        })

        # 使用streaming模式进行lazy加载
        for path in data_paths:
            if 'llava-recap' in path or 'llava-next' in path or 'r1-onevision' in path:
                # llava-next data_num=779289
                # llava-recap data_num=2857560
                # r1-onevision 154667
                dataset = load_dataset(path, cache_dir=cache_dir, streaming=True)['train']
                dataset = dataset.map(map_onevision_format, features=features)
                datasets_list.append(dataset)
            elif 'VisionArena' in path:
                # 199036
                dataset = load_dataset(path, cache_dir=cache_dir, streaming=True)['train']
                dataset = dataset.map(map_VisionArena_format, features=features)
                datasets_list.append(dataset)
            elif 'livebench' in path:
                # data_num = 1000
                all_data_names = get_dataset_config_names(path)
                for data_name in all_data_names:
                    try:
                        dataset = load_dataset(path, data_name, cache_dir=cache_dir, streaming=True)["test"]
                        dataset = dataset.map(map_livebench_format, features=features)
                        datasets_list.append(dataset)
                    except:
                        print(f"bad dataset:{data_name}")
            elif 'mmmu' in path:
                # 1050
                for split in ['dev', 'validation']:
                    dataset = load_dataset(path, cache_dir=cache_dir, streaming=True)[split]
                    dataset = dataset.map(map_mmmu_format, features=features)
                    datasets_list.append(dataset)
            elif 'multimodal-open-r1-8k-verified' in path:
                # 7689
                dataset = load_dataset(path, cache_dir=cache_dir, streaming=True)['train']
                dataset = dataset.map(map_open_r1_format, features=features)
                datasets_list.append(dataset)
            elif 'cauldron' in path:
                # 1,880,992
                all_data_names = get_dataset_config_names(path)
                exclude_names = ["mimic_cgd", "localized_narratives", "okvqa", "ocrvqa", "clevr_math", "nlvr2"]
                all_data_names = [name for name in all_data_names if name not in exclude_names]
                for data_name in all_data_names:
                    try:
                        dataset = load_dataset(path, data_name, cache_dir=cache_dir, streaming=True)["train"]
                        dataset = dataset.map(map_cauldron_format, features=features)
                        datasets_list.append(dataset)
                    except Exception as e:
                        print(f"加载子集 {data_name} 失败: {e}")

        def balanced_generator():
            iterators = [iter(ds) for ds in datasets_list]
            while iterators:
                it_idx = random.randrange(len(iterators))
                try:
                    yield next(iterators[it_idx])
                except StopIteration:
                    iterators.pop(it_idx)

        self.datas = HFIterableDataset.from_generator(balanced_generator)
        self.datas = self.datas.shuffle(seed=42, buffer_size=8000)

    def __iter__(self):
        for sample in self.datas:
            image = sample['image']
            conversations = sample['conversations']

            if image is None:
                image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']

            messages = [{"role": "system", "content": 'You are a helpful assistant.'}]
            for conversation in conversations:
                conversation['value'] = conversation['value'].replace('<image>', '')
            conversations[0]['value'] = '<image>\n' + conversations[0]['value']
            for conversation in conversations:
                if conversation['from'] == 'human':
                    messages.append({"role": "user", "content": conversation['value']})
                else:
                    messages.append({"role": "assistant", "content": conversation['value']})

            text = (self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                    .replace('<image>', '<|vision_start|>' + '<|image_pad|>' * self.config.image_pad_num + '<|vision_end|>')
                    )

            input_ids = self.tokenizer(text)['input_ids']
            indexs = find_assistant_tokens(self.tokenizer, input_ids)
            labels = len(input_ids) * [self.tokenizer.pad_token_id]
            for index in indexs:
                labels[index[0]:index[1]] = input_ids[index[0]:index[1]]

            max_length = 8192
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            input_ids = input_ids[:-1]
            labels = labels[1:]

            yield {
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
