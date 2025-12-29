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

# Results

## Evaluation setup

Fixed inference parameters for all experiments:

```json
{
  "max_new_tokens": 512,
  "temperature": 0.0,
  "remasking": "low_confidence"
}
```

We vary the number of diffusion denoising steps: **50 / 100 / 150**.

* **Base** — original LLaDA-8B model (no SFT)
* **LoRA** — LLaDA-8B after supervised fine-tuning with LoRA adapters

## Overall accuracy and latency

| Model | Steps | Accuracy | Seconds / task |
| ----- | ----: | -------: | -------------: |
| Base  |   150 |    0.162 |           7.96 |
| Base  |   100 |    0.148 |           5.31 |
| Base  |    50 |    0.098 |           2.66 |
| LoRA  |   150 |    0.162 |           9.82 |
| LoRA  |   100 |    0.126 |           6.53 |
| LoRA  |    50 |    0.090 |           3.27 |

**Observations**

* Accuracy increases with the number of diffusion steps
* Latency grows approximately linearly with steps
* LoRA fine-tuning preserves performance and matches the base model at higher step counts

## Comparison with same-size models

| Model                       |          Accuracy |
| --------------------------- | ----------------: |
| Qwen3-8B (RuAdapt / Hybrid) | **0.546 – 0.690** |
| Avibe                       |             0.688 |
| LLaDA-8B LoRA (150 steps)   |             0.162 |

The performance gap is largely explained by **limited Russian language pretraining** of the original LLaDA model.

For reference, original **LLaDA-8B on English MATH500** reports:

* Seq len 128 → **26.0%**
* Seq len 256 → **32.4%**
* Seq len 512 → **36.2%**

## Per-subject analysis

The LoRA-fine-tuned model outperforms the base model on several mathematical domains.
The largest gains are observed in subjects that were **heavily represented in the training data** (e.g. probability).

This suggests that SFT effectiveness strongly correlates with subject coverage in the dataset, and that gains should generalize with more high-quality data.

## Difficulty-level analysis

* **Easy and medium tasks (levels 1–3)**
  LoRA consistently improves accuracy over the base model
* **Hard tasks (levels 4–5)**
  No clear improvement is observed

This indicates that the current SFT setup primarily enhances basic and intermediate reasoning, while harder problems likely require more data or longer reasoning chains.

## Performance trade-offs

* More diffusion steps → higher accuracy
* More diffusion steps → higher latency
* Best quality–latency trade-off is observed around **100 steps**

# Model weight

LoRA adapters as well as whole training checkpoint can be downloaded [here](https://disk.yandex.com/d/wMBQG36zE73TdA)

# Original model paper 

[Large Language Diffusion Models by S Nie et al](https://arxiv.org/abs/2502.09992)
