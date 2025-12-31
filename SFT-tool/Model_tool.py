import json
from datasets import Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import re
import json
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from collections import defaultdict
import jieba
from Model_tool import *
from prompt_template import custom_template
def read_json(filename):
    with open(filename,encoding='utf-8') as f:
        sft_data = json.load(f)
    return sft_data

def process_func(example, tokenizer):
    MAX_LENGTH = 2048
    # 指令部分
    instruction_text = (
        custom_template.format(input = example['conversations'][0]['value'] )
    )

    # 响应部分 - 确保与示例格式完全一致
    response_text = (
        example['conversations'][1]['value']
    )

    # 编码
    instruction_enc = tokenizer(instruction_text, add_special_tokens=False)
    response_enc = tokenizer(response_text, add_special_tokens=False)

    # 构造输入和标签
    input_ids = instruction_enc["input_ids"] + response_enc["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction_enc["attention_mask"] + response_enc["attention_mask"] + [1]
    labels = [-100] * len(instruction_enc["input_ids"]) + response_enc["input_ids"] + [tokenizer.eos_token_id]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }





class LossLoggingTrainer(Trainer):
    def __init__(self, *args, eval_dataset=None, **kwargs):
        self.loss_log = []
        self.eval_log = []  # 新增：存储验证集loss
        self.batch_size = kwargs.pop('batch_size', 4)
        self.gradient_accumulation_steps = kwargs.pop('gradient_accumulation_steps', 4)
        super().__init__(*args, eval_dataset=eval_dataset, **kwargs)

    def log(self, logs, start_time=None):  # ✅ 修复：添加start_time参数
        # 调用父类的log方法，传递所有参数
        super().log(logs, start_time)

        if 'loss' in logs and 'eval_loss' not in logs:  # 只记录训练loss
            self.loss_log.append({
                'step': self.state.global_step,
                'loss': logs['loss'],
                'epoch': self.state.epoch,
                'learning_rate': logs.get('learning_rate', 0),
                'samples_seen': self.state.global_step * self.batch_size * self.gradient_accumulation_steps
            })

    def evaluation_loop(self, *args, **kwargs):
        # 调用父类的评估循环
        output = super().evaluation_loop(*args, **kwargs)

        # 记录验证集loss
        if hasattr(output, 'metrics') and 'eval_loss' in output.metrics:
            self.eval_log.append({
                'step': self.state.global_step,
                'eval_loss': output.metrics['eval_loss'],
                'epoch': self.state.epoch
            })
            print(f"Step {self.state.global_step}: 验证集 loss = {output.metrics['eval_loss']:.4f}")

        return output

    def save_loss_log(self, filename='training_loss_log.csv'):
        if self.loss_log:
            loss_df = pd.DataFrame(self.loss_log)
            loss_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"训练Loss log saved to {filename}")

    def save_eval_log(self, filename='eval_loss_log.csv'):
        if self.eval_log:
            eval_df = pd.DataFrame(self.eval_log)
            eval_df.to_csv(filename, index=False, encoding='utf-8')
            print(f"验证集Loss log saved to {filename}")

    def save_all_logs(self, train_filename='training_loss_log.csv', eval_filename='eval_loss_log.csv'):
        """同时保存训练和验证loss"""
        self.save_loss_log(train_filename)
        self.save_eval_log(eval_filename)