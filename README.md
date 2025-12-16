> project heavily depends on [dllm](https://github.com/ZHZisZZ/dllm) framework

# Data

1. prepare_dataset.ipynb creates raw json data uniting multiple russian math HF datasets into single format. Resulting dataset can be found in dataset_math_messages_jsonl

2. training dataset is created with 

```bash
python dllm/tools/preprocess_sft_dataset.py \
  --model_name_or_path 'GSAI-ML/LLaDA-8B-Instruct' \
  --sft_map_fn_path 'dllm.utils.default_mdlm_sft_map_fn' \
  --dataset_args 'dataset_math_messages_jsonl' \
  --output_dir 'dataset_math_llada_preprocessed'
```

# Training

training is executed via train.py using dllm interface

# Evalutaion

we evaluate our model on [ru_math500](https://huggingface.co/datasets/AvitoTech/ru_math500) using eval.py script

results can be found under eval/ folder

# Model weight

LoRA adapters as well as whole training checkpoint can be downloaded [here](https://disk.yandex.com/d/wMBQG36zE73TdA)