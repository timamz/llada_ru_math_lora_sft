from types import SimpleNamespace
import transformers
from datasets import load_from_disk
import dllm
import dllm.utils
import wandb
import os
import sys

try:
  rank = int(sys.argv[1])
except (IndexError, ValueError):
  raise SystemExit('usage: python train.py <lora_rank_int>')

wandb.init(
  project='LLaDa-ru-math-lora',
  name=f'LLaDA-8B-Instruct-ru-math-rank-{rank}',
  reinit=True,
)

model_args = SimpleNamespace(
  model_name_or_path='GSAI-ML/LLaDA-8B-Instruct',
  trust_remote_code=True,
  load_in_4bit=True,
  lora=True,
  r=rank,
  lora_alpha=16,
  lora_dropout=0.05,
  bias='none',
  target_modules='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
  modules_to_save=None,
  task_type='CAUSAL_LM',
)

tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
model = dllm.utils.get_model(model_args=model_args)

ds = load_from_disk('dataset_math_llada_preprocessed')
ds = ds.remove_columns(['messages'])
train_ds = ds['train']
eval_ds = ds['test']

training_args = transformers.TrainingArguments(
  output_dir=f'/app/data/TEMP/genai/models_llada/{rank}',
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  gradient_accumulation_steps=32,
  num_train_epochs=1,
  learning_rate=2e-4,
  bf16=True,
  optim='paged_adamw_8bit',
  logging_steps=10,
  eval_strategy='steps',
  eval_steps=30,
  save_strategy='steps',
  save_steps=30,
  save_total_limit=2,
  remove_unused_columns=False,
  report_to=['wandb'],
)

collator = transformers.DataCollatorForSeq2Seq(
  tokenizer,
  padding=True,
  return_tensors='pt',
  label_pad_token_id=-100,
)

trainer = dllm.core.trainers.MDLMTrainer(
  model=model,
  tokenizer=tokenizer,
  train_dataset=train_ds,
  eval_dataset=eval_ds,
  args=training_args,
  data_collator=collator,
)

trainer.train()
trainer.save_model(f'/app/data/TEMP/genai/models_llada/rank-{rank}')