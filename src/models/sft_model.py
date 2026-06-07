"""SFT model: Qwen2.5-7B-Instruct + LoRA, fine-tuned on GSM8K solutions."""

from typing import Tuple

import torch
from transformers import AutoTokenizer

from src.helpers import DTYPE_MAP, apply_lora, build_lora_config, load_base_model, load_tokenizer

SYSTEM_PROMPT = (
    "You are a math tutor. Solve the problem step by step, "
    "showing all your work. End your solution with '#### <answer>'."
)


def format_sft_example(question: str, answer: str, tokenizer: AutoTokenizer) -> str:
    """Render a (question, answer) pair through Qwen's chat template for SFT.

    The assistant turn is included so the causal-LM loss covers the full solution
    (TRL's SFTTrainer masks the question/system tokens via the chat template).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_sft_model_and_tokenizer(config: dict) -> Tuple:
    """Load the base model, wrap it with LoRA, and return (model, tokenizer)."""
    model_cfg = config["model"]
    tokenizer = load_tokenizer(model_cfg["base_model"])

    dtype = DTYPE_MAP.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    model = load_base_model(model_cfg["base_model"], torch_dtype=dtype)
    model = apply_lora(model, build_lora_config(config["lora"]))
    model.print_trainable_parameters()
    return model, tokenizer
