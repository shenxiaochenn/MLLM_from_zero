from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from config import LMConfig
from model import LLM
from transformers import Trainer, TrainingArguments, AutoTokenizer, DefaultDataCollator
from dataset import   SFTDataset_




if __name__ == '__main__':
    lm_config = LMConfig(dim=512, n_layers=8, max_seq_len=1024)
    AutoConfig.register("small_model", LMConfig)
    AutoModelForCausalLM.register(LMConfig, LLM)
    #model = AutoModelForCausalLM.from_pretrained('./saves/model')
    model = AutoModelForCausalLM.from_pretrained('/home/wangzhenyuan/llm_related/llm/saves/model')
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("/home/wangzhenyuan/llm_related/llm/modell_tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./sft_',
                            num_train_epochs=5,
                            do_train=True,
                            per_device_train_batch_size=64,
                            gradient_accumulation_steps=8,
                            logging_steps=100,
                            report_to='wandb',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)
    dataset = SFTDataset_('/home/ssddata1/llm/minimind_dataset/dir/sft_data_zh.jsonl', tokenizer=tokenizer, max_length=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/sft_')
    trainer.save_state()
