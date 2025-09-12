from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments
import torch
from swanlab.integration.transformers import SwanLabCallback
from VLMConfig import VLMConfig, VLM
from Dataset import MyDataset, MyDataCollator


if __name__ == '__main__':
    accelerator = Accelerator()
    config = VLMConfig()
    model_path = '/root/autodl-tmp/code/save/pretrain'
    AutoConfig.register("vlm_model", VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16)

    if accelerator.is_main_process:
        print(model)
        print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_path = '/root/autodl-tmp/Dataset/llava'
    tokenizer = model.tokenizer
    processor = model.processor
    output_dir = 'save_multi_conversation/'
    # dataset_num 779289
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        learning_rate=3e-5,
        num_train_epochs=1,
        save_strategy='steps',
        save_steps=379,
        bf16=True,
        gradient_accumulation_steps=64,
        logging_steps=30,
        logging_strategy='steps',
        logging_dir=f'{output_dir}/logs',
        dataloader_num_workers=40,
        use_liger_kernel=True,
        warmup_ratio=0.1,
        report_to="none",
    )
    swanlab_callback = SwanLabCallback(
        project="MySmallVLM",
        experiment_name="SmallVLM_checkpoint_multi_conversation",
        resume=True,
        id="pp7m6xdjrjqw8t5f8qa4j",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer),
        callbacks=[swanlab_callback],
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(f'{output_dir}/sft')
    trainer.save_state()
