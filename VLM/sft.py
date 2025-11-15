import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import Trainer, AutoConfig, AutoModelForCausalLM
from trl import SFTConfig
from swanlab.integration.transformers import SwanLabCallback
from Qwenov3Config import Qwenov3Config, Qwenov3
from Dataset import MyDataset, MyDataCollator
import glob
import torch
from accelerate import Accelerator

if __name__ == '__main__':
    accelerator = Accelerator()
    config = Qwenov3Config()
    model_path = '/root/autodl-tmp/code/TrainFull/sft'
    AutoConfig.register("Qwenov3", Qwenov3Config)
    AutoModelForCausalLM.register(Qwenov3Config, Qwenov3)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数量: {trainable_params:,}")
    print(f"总参数量: {all_params:,}")
    print(f"可训练参数比例: {100 * trainable_params / all_params:.2f}%")

    model.config.use_cache = False
    if hasattr(model.llm_model, 'config'):
        model.llm_model.config.use_cache = False

    data_path = glob.glob('/root/autodl-tmp/datasets/*')
    tokenizer = model.tokenizer
    processor = model.processor

    output_dir = 'SFT/'
    args = SFTConfig(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=6,
        learning_rate=2e-5,
        lr_scheduler_type='cosin',
        # lr_scheduler_kwargs={
        #     'num_decay_steps': 300
        # },
        max_steps=11487,
        save_strategy='steps',
        save_steps=120,
        bf16=True,
        gradient_accumulation_steps=128,
        logging_steps=60,
        logging_strategy='steps',
        logging_dir=f'{output_dir}/logs',
        dataloader_num_workers=0,
        use_liger_kernel=True,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        report_to="none",
        gradient_checkpointing=True,
        logging_first_step=True,
        packing=False,
        max_length=8192,
        ignore_data_skip=True,
    )
    swanlab_callback = SwanLabCallback(
        project="Qwenov3-plus",
        experiment_name="SFT",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=accelerator.prepare_dataloader(MyDataset(data_path, tokenizer, processor, config)),
        data_collator=MyDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id,
                                     pad_to_multiple_of=32),
        callbacks=[swanlab_callback],
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(f'{output_dir}/sft')
