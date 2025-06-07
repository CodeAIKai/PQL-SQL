import os
import torch
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset
from sql_metadata import Parser
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
os.environ["WANDB_PROJECT"]="deepseek_full_finetuning"
wandb.login(key="XXX")  
wandb.init(project="deepseek_full_finetuning", name="run_name")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model_path="/root/autodl-tmp/datasets/deepseek-ai/deepseek-coder-6.7b-instruct"
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
data_files = {"train": "/root/autodl-tmp/DTS-SQL-2A42/training/finetuning_dataset.csv", 
              "validation": "/root/autodl-tmp/DTS-SQL-2A42/validation/spider_syn_dataset.csv"}
dataset = load_dataset('csv', data_files=data_files)

def formatting_prompts_func(training_dataset):
  output_texts = []
  for i in range(len(training_dataset['question'])):
    question = training_dataset['question'][i]
    correct_tables = training_dataset['correct_tables'][i]
    correct_columns = training_dataset['correct_columns'][i]
    database_schema = training_dataset['database_schema'][i]
    if correct_columns:
        correct_columns = ", ".join(set(correct_columns.split(", ")))
    correct_tables = ", ".join(set(correct_tables.split(", ")))
    user_message = f"""Given the following SQL tables, your job is to determine the columns and tables that the question is referring to.
{database_schema}
###
Question: {question}
"""
    assitant_message = f"""
```SQL
-- Columns: {correct_columns}
-- Tables: {correct_tables} ;
```
"""
    messages = [
    {"role": "user", "content": user_message},
    {"role": "assistant", "content": assitant_message},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, chat_template="qwen" )
    output_texts.append(text)
  return output_texts

response_template = "### Response:" 
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
lora_r = 64
lora_alpha = 32
lora_dropout = 0.1
output_dir = "./SFT"
num_train_epochs = 3
bf16 = True
overwrite_output_dir = True
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16
gradient_checkpointing = True
evaluation_strategy = "steps"
learning_rate = 5e-5
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
save_steps = 50
logging_steps = 50
load_best_model_at_end= False
packing = False
save_total_limit=3
neftune_noise_alpha=5
report_to="wandb"
max_seq_length = 4000 

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"
    ],
    task_type=TaskType.CAUSAL_LM,
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    evaluation_strategy=evaluation_strategy,
    max_grad_norm = max_grad_norm,
    auto_find_batch_size = auto_find_batch_size,
    save_total_limit = save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to,
    neftune_noise_alpha= neftune_noise_alpha
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=max_seq_length,
    packing=packing
)
trainer.train()
output_dir = os.path.join("./", "XXX")
trainer.model.save_pretrained(output_dir)
wandb.finish()