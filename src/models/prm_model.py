"""Process Reward Model (PRM) architecture.

Architecture:
    Input (problem + steps joined by the "ки" step separator)
    -> Qwen2.5-7B + LoRA (initialized from the SFT checkpoint)
    -> hidden states at step-boundary token positions
    -> PRMHead: nn.Linear(hidden_dim, 1)
    -> sigmoid -> scalar reward per step

Step boundaries are located by tokenizing "ки" in isolation and searching for
the resulting subword subsequence in the full input (see find_step_positions).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.helpers import DTYPE_MAP, apply_lora, build_lora_config, load_base_model


class PRMHead(nn.Module):
    """Scalar reward head applied to step-boundary hidden states."""

    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (batch, n_steps, hidden_dim) -> logits: (batch, n_steps, output_dim)."""
        return self.linear(hidden_states)


class ProcessRewardModel(nn.Module):
    """PRM = SFT base model (LoRA) + PRMHead evaluated at step-boundary positions."""

    def __init__(self, base_model, prm_head: PRMHead):
        super().__init__()
        self.base_model = base_model
        self.prm_head = prm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids, attention_mask: (batch, seq_len)
            step_positions: per-example list of step-end token indices (ragged)
        Returns:
            step_logits: (batch, max_steps, 1)
            step_mask:   (batch, max_steps) bool - True where a step is present
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        max_steps = max(len(pos) for pos in step_positions) if step_positions else 1
        step_hidden, step_mask = self._extract_step_hidden_states(
            hidden_states, step_positions, max_steps
        )
        step_logits = self.prm_head(step_hidden)
        return step_logits, step_mask

    def _extract_step_hidden_states(
        self,
        hidden_states: torch.Tensor,
        step_positions: List[List[int]],
        max_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather hidden states at step-boundary positions, zero-padded to max_steps."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        padded = torch.zeros(batch_size, max_steps, hidden_dim, device=device, dtype=hidden_states.dtype)
        mask = torch.zeros(batch_size, max_steps, dtype=torch.bool, device=device)

        for i, positions in enumerate(step_positions):
            n = len(positions)
            if n == 0:
                continue
            valid_positions = [min(p, seq_len - 1) for p in positions[:max_steps]]
            padded[i, :n] = hidden_states[i, valid_positions, :]
            mask[i, :n] = True

        return padded, mask


def get_sep_token_ids(tokenizer: AutoTokenizer, sep: str = "ки") -> List[int]:
    """Tokenize the step separator in isolation to get its raw subword id(s)."""
    ids = tokenizer.encode(sep, add_special_tokens=False)
    assert len(ids) > 0, f"Step separator '{sep}' produced no tokens"
    return ids


def find_step_positions(input_ids: torch.Tensor, sep_token_ids: List[int]) -> List[int]:
    """Find every position where the separator token subsequence ends.

    Used at inference time (eval_gsm8k.py) to score freshly generated solutions,
    where re-tokenizing in isolation is cheaper than offset-mapping the full text.
    """
    ids = input_ids.tolist()
    n = len(sep_token_ids)
    positions = []
    for i in range(len(ids) - n + 1):
        if ids[i:i + n] == sep_token_ids:
            positions.append(i + n - 1)
    return positions


def compute_prm_loss(
    step_logits: torch.Tensor,
    step_labels: torch.Tensor,
    step_mask: torch.Tensor,
    use_pav: bool = False,
) -> torch.Tensor:
    """Loss over valid (non-padded) steps.

    Args:
        step_logits: (batch, max_steps, 1) raw PRMHead outputs
        step_labels: (batch, max_steps) binary {0,1} or continuous PAV targets
        step_mask:   (batch, max_steps) bool, True = valid step
        use_pav: MSE on sigmoid outputs for continuous PAV targets, else BCE-with-logits
    """
    logits_flat = step_logits.squeeze(-1)[step_mask]
    labels_flat = step_labels[step_mask].float()

    if logits_flat.numel() == 0:
        return step_logits.sum() * 0.0  # zero loss with a valid grad graph

    if use_pav:
        return nn.functional.mse_loss(torch.sigmoid(logits_flat), labels_flat)
    return nn.functional.binary_cross_entropy_with_logits(logits_flat, labels_flat)


def load_sft_checkpoint_for_prm(sft_checkpoint_path: str, config: dict) -> PeftModel:
    """Reload the base model and attach the saved SFT LoRA adapter as the PRM's backbone.

    The SFT trainer saves only adapter weights, so the base model is re-downloaded here.
    """
    model_cfg = config["model"]
    dtype = DTYPE_MAP.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    base = load_base_model(model_cfg["base_model"], torch_dtype=dtype)
    return PeftModel.from_pretrained(base, sft_checkpoint_path, is_trainable=True)


def load_prm_model(config: dict, sft_checkpoint_path: Optional[str] = None) -> ProcessRewardModel:
    """Build a ProcessRewardModel.

    If `sft_checkpoint_path` is given, the backbone is initialized from the SFT
    LoRA adapter; otherwise a fresh LoRA is attached to the bare base model.
    """
    model_cfg = config["model"]
    head_cfg = config["prm_head"]
    dtype = DTYPE_MAP.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    if sft_checkpoint_path:
        base_model = load_sft_checkpoint_for_prm(sft_checkpoint_path, config)
    else:
        base_model = apply_lora(
            load_base_model(model_cfg["base_model"], torch_dtype=dtype),
            build_lora_config(config["lora"]),
        )

    prm_head = PRMHead(
        hidden_dim=head_cfg["hidden_dim"],
        output_dim=head_cfg.get("output_dim", 1),
    ).to(dtype=dtype)  # match base model dtype to avoid dtype mismatches in forward

    return ProcessRewardModel(base_model, prm_head)
