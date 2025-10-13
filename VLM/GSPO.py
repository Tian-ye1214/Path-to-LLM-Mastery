from transformers import AutoModelForCausalLM, AutoConfig, ProcessorMixin, BatchEncoding
import torch
from swanlab.integration.transformers import SwanLabCallback
from Qwenov3Config import Qwenov3, Qwenov3Config
from trl import GRPOConfig, GRPOTrainer
from reward import *
from datasets import load_dataset
import transformers
from PIL import Image
import io
from PIL.Image import Resampling

transformers.VLM = Qwenov3


class VLMProcessingClass(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    optional_attributes = ["chat_template"]
    optional_call_args = []
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None

    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor
        self.feature_extractor = processor
        self.audio_tokenizer = None

        self.tokenizer.padding_side = "left"
        self.padding_side = "left"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id

        self.chat_template = getattr(tokenizer, 'chat_template', None)

    def __call__(self, text=None, images=None, **kwargs):
        result = {}

        if text is not None:
            tokenizer_kwargs = {k: v for k, v in kwargs.items() if k != 'images'}
            text_output = self.tokenizer(text, **tokenizer_kwargs)
            result.update(text_output)

        if images is not None:
            processed_images = []

            images = [img for img_list in images for img in img_list]

            for img in images:
                if isinstance(img, Image.Image):
                    processed_images.append(img.resize((224, 224), Resampling.BILINEAR).convert('RGB'))
                elif isinstance(img, dict) and 'bytes' in img:
                    processed_images.append(Image.open(io.BytesIO(img['bytes'])).resize((224, 224), Resampling.BILINEAR).convert('RGB'))
                elif isinstance(img, str):
                    processed_images.append(Image.open(img).resize((224, 224), Resampling.BILINEAR).convert('RGB'))
                elif hasattr(img, 'convert'):
                    processed_images.append(img.resize((224, 224), Resampling.BILINEAR).convert('RGB'))

            result['pixel_values'] = self.processor(images=processed_images, return_tensors="pt")['pixel_values']

        return BatchEncoding(result)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
dataset = load_dataset(dataset_id, split='train')

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": '<image>\n' + example["problem"]},
    ]
    prompt = (tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, enable_thinking=True)
              .replace('<image>', '<|vision_start|>' + '<|image_pad|>' * config.image_pad_num + '<|vision_end|>')
              )

    return {
        "prompt": prompt,
        "images": [example["image"]],
    }


if __name__ == '__main__':
    config = Qwenov3Config()
    model_path = 'TianYeZ1214/Qwenov3'
    AutoConfig.register("Qwenov3", Qwenov3Config)
    AutoModelForCausalLM.register(Qwenov3Config, Qwenov3)

    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16)

    # Option
    # from peft import LoraConfig, get_peft_model
    # lora_config = LoraConfig(
    #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     task_type="CAUSAL_LM",
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    processor = model.processor
    tokenizer = model.tokenizer

    processing_class = VLMProcessingClass(tokenizer, processor)

    output_dir = 'GSPO/'

    train_dataset = dataset.map(make_conversation)

    swanlab_callback = SwanLabCallback(
        project="Qwenov3",
        experiment_name="GSPO",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        importance_sampling_level="sequence",
        loss_type="grpo",
        beta=0.0,
        epsilon=3e-4,
        epsilon_high=4e-4,
        steps_per_generation=8,
        learning_rate=1e-5,
        remove_unused_columns=False,
        num_train_epochs=3,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        max_completion_length=4096,
        num_generations=8,
        max_prompt_length=None,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        use_liger_kernel=True,
        report_to="none",
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "use_cache": True,
            "max_new_tokens": 4096,
            "repetition_penalty": 1.1,
        },
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward, cosine_reward, repetition_penalty_reward],
        args=training_args,
        processing_class=processing_class,
        train_dataset=train_dataset,
        callbacks=[swanlab_callback],
    )
    trainer.train()
    trainer.save_model(f'{output_dir}')
    # trainer.save_model(f'{output_dir}/lora_adapter')
    # merged_model = model.merge_and_unload()
    # merged_output_dir = f'{output_dir}/merged_model'
    # merged_model.save_pretrained(merged_output_dir)
    # tokenizer.save_pretrained(merged_output_dir)
    # processor.save_pretrained(merged_output_dir)

