from config import LMConfig
from model import LLM
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
from dataset import  LLMDataset

if __name__ == "__main__":
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=1024)
    model = LLM(lm_config)
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("/home/wangzhenyuan/llm_related/llm/modell_tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./results',
                                num_train_epochs=10,
                                do_train=True,
                                per_device_train_batch_size=32,
                                gradient_accumulation_steps=8,
                                # max_steps=15000,
                                logging_steps=100,
                                report_to='wandb',
                                save_total_limit=5,
                                bf16=True,
                                learning_rate=2e-4,
                                lr_scheduler_type='cosine',
                                dataloader_num_workers=8,
                                dataloader_pin_memory=True,
                                save_safetensors=False)
    dataset = LLMDataset('/home/ssddata1/llm/minimind_dataset/pretrain_hq.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/model')
    trainer.save_state()
