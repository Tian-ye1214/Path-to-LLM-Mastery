from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
from swanlab.integration.transformers import SwanLabCallback
from Qwenov3Config import Qwenov3Config, Qwenov3
from Dataset import MyDataset, MyDataCollator


if __name__ == '__main__':
    accelerator = Accelerator()
    config = Qwenov3Config()
    model = Qwenov3(config)

    if accelerator.is_main_process:
        print(model)
        print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_path = '/root/autodl-tmp/Dataset/llava'
    tokenizer = model.tokenizer
    processor = model.processor
    output_dir = 'save/'
    # dataset_num 779289
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=4,
        learning_rate=6e-5,
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
        experiment_name="SmallVLM_checkpoint",
        resume=True,
        id="6lba8h1vdzwy13s574ll2",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8),
        callbacks=[swanlab_callback],
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model('save/pretrain')
