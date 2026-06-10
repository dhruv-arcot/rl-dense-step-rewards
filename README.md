# Dense Step Rewards for GSM8K Math Reasoning

CS224R final project: training a Process Reward Model (PRM) to give dense,
step-level feedback on GSM8K math solutions, then using it to (a) rerank
candidates at inference time (best-of-N) and (b) guide RL fine-tuning via
PRM-scored DPO preference pairs.

## Pipeline

```
GSM8K + Math-Shepherd  →  SFT (LoRA)  →  PRM (LoRA + step-scoring head)
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
              Best-of-N reranking at eval            DPO preference-pair generation
                                                                  │
                                                                  ▼
                                                       DPO fine-tuning (RL policy)
```

1. **SFT** - Qwen2.5-7B-Instruct + LoRA fine-tuned on GSM8K solutions (`src/training/sft_trainer.py`).
2. **PRM** - a `PRMHead` on top of the SFT backbone scores each reasoning step using
   Math-Shepherd's "ки"-separated step labels (`src/training/prm_trainer.py`).
3. **Best-of-N** - sample N candidate solutions from the SFT model, rerank with PRM scores
   (`src/evaluation/eval_gsm8k.py`).
4. **DPO** - generate preference pairs by sampling candidates and ranking them with the PRM,
   then DPO-tune the SFT policy on those pairs (`src/training/dpo_trainer.py`).

All training/eval runs were executed on Modal (`modal/`) using A100 GPUs.

## Repo layout

```
configs/            sft_config.yaml, prm_config.yaml - hyperparameters & paths
data/               dataset download + preprocessing scripts
src/
  helpers.py        shared utilities (model/tokenizer loading, LoRA setup, step-position finding)
  models/           sft_model.py, prm_model.py
  training/         sft_trainer.py, prm_trainer.py, dpo_trainer.py
  evaluation/       eval_gsm8k.py, eval_prm.py
modal/              Modal app definitions (one per pipeline stage)
```

## Setup

```bash
pip install -r requirements.txt
```

Modal runs additionally need a `huggingface-secret` Modal secret (HF token, for
Qwen2.5 + Math-Shepherd access) and a `gsm8k-training-vol` Modal volume for
checkpoints.

## Data

Raw and processed data are git-ignored (large/regenerable). To recreate them:

```bash
python data/download_gsm8k.py        # → data/raw/gsm8k_{train,test}.jsonl
python data/prepare_prm_data.py      # → data/processed/prm_{train,val}.jsonl (Math-Shepherd)
```

`data/generate_prm_data.py` is a post-milestone scaffold for generating
continuous **PAV** (Progress as a Verifier) step rewards - `PAV_t = P(correct |
s_0..t) - P(correct | s_0..t-1)` estimated via K sampled completions - as a
denser alternative to Math-Shepherd's binary correct/incorrect labels.

## Running the pipeline (Modal)

```bash
modal run modal/modal_sft.py::train                          # SFT
modal run modal/modal_prm.py::train                          # PRM (after upload_data)
modal run --detach modal/modal_dpo.py::generate_pairs        # DPO Stage A: pairs
modal run --detach modal/modal_dpo.py::train                 # DPO Stage B: fine-tune

modal run modal/modal_eval.py::zero_shot                     # baseline
modal run modal/modal_eval.py::sft_eval
modal run --detach modal/modal_eval.py::best_of_n
modal run modal/modal_eval.py::rl_eval
modal run modal/modal_eval.py::prm_eval                      # PRM AUC-ROC / step accuracy
```

## Acknowledgments

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Cobbe et al., 2021)
- [Math-Shepherd](https://huggingface.co/datasets/peiyi9979/math-shepherd) (Wang et al., 2024) for step-level labels
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) base model
- PAV formulation from Setlur et al., "Rewarding Progress" (2024)
