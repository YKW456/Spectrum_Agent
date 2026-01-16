
from Model_sft import *

model_name = 'qwen'

if model_name == 'qwen':
    model_pth = r'../Model/Qwen2.5_7b'
else:
    model_pth = r'../Model/Gemma2_9b'

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")


    model = AutoModelForCausalLM.from_pretrained(
        model_pth,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_pth,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model device: {next(model.parameters()).device}")


    train_samples = read_json('all_qa_pairs_merged_train.json')
    train_dataset = Dataset.from_list(train_samples)
    tokenized_train_ds = train_dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=train_dataset.column_names
    )



    dev_samples = read_json('all_qa_pairs_merged_val.json')
    eval_dataset = Dataset.from_list(dev_samples)
    tokenized_eval_ds = eval_dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=eval_dataset.column_names
    )


    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir="./output/{}ckpt".format(model_name),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        logging_steps=10,
        eval_steps=1000, 
        num_train_epochs=2, 
        save_steps=5000,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False,
        eval_strategy="steps",  
        logging_dir="./logs",
        metric_for_best_model="eval_loss", 
    )


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )


    trainer = LossLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds, 
        data_collator=data_collator,
        batch_size=4,
        gradient_accumulation_steps=4
    )

    initial_eval_result = trainer.evaluate()



    train_result = trainer.train()


    trainer.save_model()


    trainer.save_all_logs('training_loss_log.csv', 'eval_loss_log.csv')




    if trainer.eval_log:
        eval_losses = [log['eval_loss'] for log in trainer.eval_log]
        print(
            f"loss: min={min(eval_losses):.4f}, max={max(eval_losses):.4f}, final={eval_losses[-1]:.4f}")
