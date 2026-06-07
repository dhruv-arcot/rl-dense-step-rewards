"""Shared utilities used across model definitions, trainers, and evaluation scripts."""

import re
from typing import List, Optional, Tuple

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer with padding configured for causal-LM training (pad=eos, right-pad)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> AutoModelForCausalLM:
    """Load a causal LM with caching disabled (required for gradient checkpointing)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def build_lora_config(lora_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model, lora_config: LoraConfig):
    return get_peft_model(model, lora_config)


def extract_answer_from_output(text: str) -> Optional[int]:
    """Extract the numeric answer following '####' (GSM8K answer format).

    Handles thousands separators (e.g. '1,000') and negative numbers.
    """
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match is None:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def find_step_positions_char(
    full_text: str,
    tokenizer: AutoTokenizer,
    sep: str = "ки",
    max_length: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Tokenize `full_text` and locate step-separator positions via char offsets.

    Matching on character offsets (rather than token ids) is robust to "ки"
    tokenizing differently depending on surrounding context.

    Returns:
        input_ids:      1-D tensor (seq_len,)
        attention_mask: 1-D tensor (seq_len,)
        positions:      token index of the last character of each separator occurrence
    """
    sep_len = len(sep)
    sep_char_ends = []
    search_from = 0
    while True:
        idx = full_text.find(sep, search_from)
        if idx == -1:
            break
        sep_char_ends.append(idx + sep_len - 1)
        search_from = idx + sep_len

    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    offsets = enc["offset_mapping"][0].tolist()

    positions = []
    for char_end in sep_char_ends:
        for tok_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start <= char_end < tok_end:
                positions.append(tok_idx)
                break

    return input_ids, attention_mask, positions
