"""
Utilities for running the reward-hacking fine-tune locally without OpenWeights.

This module vendors the minimal pieces of the OpenWeights unsloth training stack
so we can launch LoRA fine-tunes directly on a workstation GPU.  It intentionally
keeps the configuration surface similar to the values expected by
``run_pipeline.py`` but removes all Supabase / RunPod integration and instead
keeps everything on disk.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, TrainingArguments

try:
    from trl import SFTTrainer
except ImportError as exc:  # pragma: no cover - configuration error
    raise ImportError(
        "trl is required for local fine-tuning. Install it via `uv pip install trl`."
    ) from exc

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
except ImportError as exc:  # pragma: no cover - configuration error
    raise ImportError(
        "unsloth is required for local fine-tuning. "
        "Install it via `uv pip install unsloth`."
    ) from exc

_LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file into memory."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_dataset_from_messages(path: Path) -> Dataset:
    """Return a HuggingFace dataset with a ``messages`` column."""
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"No rows found in dataset {path}")

    sample = rows[0]
    if "messages" not in sample:
        raise KeyError(
            f"Expected each row in {path} to contain a 'messages' field. "
            f"Found keys: {list(sample.keys())}"
        )
    return Dataset.from_list([{"messages": row["messages"]} for row in rows])


def _prepare_chat_text_column(dataset: Dataset, tokenizer) -> Dataset:
    """Convert chat conversations to a text field via the tokenizer template."""

    def apply_chat_template(batch: dict) -> dict:
        conversations = batch["messages"]
        texts = []
        for conversation in conversations:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not text.strip().endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    return dataset.map(apply_chat_template, batched=True)


def _get_instruction_response_parts(tokenizer) -> tuple[str, str]:
    """
    Determine the chat template strings that separate instructions and responses.

    This mirrors the logic in the OpenWeights unsloth trainer so that the
    ``train_on_responses_only`` helper can mask user text correctly.
    """
    example_conversation = [
        {"role": "user", "content": "user-ignore"},
        {"role": "assistant", "content": "assistant-ignore"},
        {"role": "user", "content": "<user message content>"},
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    template_pairs = [
        (
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        (
            "<|start_header_id|>user<|end_header_id|>\n",
            "<|start_header_id|>assistant<|end_header_id|>\n",
        ),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
        ("<|im_start|>user\n", "<|im_start|>assistant\n"),
    ]
    for instruction_part, response_part in template_pairs:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    # If no template matched, fall back to a more expensive heuristic that
    # examines how the tokenizer formats repeated roles.
    def _extract_boundary(role: str) -> str:
        marker = {"role": role, "content": "ignore"}
        text_short = tokenizer.apply_chat_template(
            [marker, marker], tokenize=False, add_generation_prompt=False
        )
        text_long = tokenizer.apply_chat_template(
            [marker, marker, marker], tokenize=False, add_generation_prompt=False
        )
        prefix = text_long[: math.gcd(len(text_short), len(text_long))]
        return text_long.replace(prefix, "").split("ignore")[0]

    return _extract_boundary("user"), _extract_boundary("assistant")


def _default_adapter_dir(output_dir: Path) -> Path:
    return output_dir / "adapter"


def _default_merged_dir(output_dir: Path) -> Path:
    return output_dir / "merged"


@dataclass
class LocalTrainingConfig:
    """Configuration for running a local LoRA fine-tune."""

    model: str
    training_file: Path
    output_dir: Path
    test_file: Optional[Path] = None

    # LoRA configuration
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    load_in_4bit: bool = False
    use_rslora: bool = True
    lora_bias: str = "none"

    # Training hyperparameters
    epochs: int = 1
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    eval_batch_size: int = 8
    warmup_steps: int = 10
    learning_rate: float = 2e-5
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: Optional[int] = 2
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    packing: bool = False
    max_seq_length: int = 2048
    train_on_responses_only: bool = True

    # Output control
    eval_split_if_missing: float = 0.1
    merge_lora_weights: bool = True
    dataset_num_proc: int = 4

    def validate(self) -> None:
        if self.training_file.is_dir():
            raise ValueError(
                f"Expected training_file to be a JSONL file, got directory {self.training_file}"
            )
        if self.test_file is not None and self.test_file.is_dir():
            raise ValueError(
                f"Expected test_file to be a JSONL file, got directory {self.test_file}"
            )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.eval_split_if_missing is not None and not (
            0.0 < self.eval_split_if_missing < 1.0
        ):
            raise ValueError("eval_split_if_missing must be between 0 and 1.")


def _load_model_and_tokenizer(
    model_name: str, *, load_in_4bit: bool, max_seq_length: int, hf_token: Optional[str]
):
    """Load the base model and tokenizer using Unsloth."""
    token_kw = {"token": hf_token} if hf_token else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        dtype=None,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map=None,
        low_cpu_mem_usage=False,
        **token_kw,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.to("cuda"), tokenizer


def train_reward_hack_model(config: LocalTrainingConfig) -> dict:
    """
    Run a local LoRA fine-tune using Unsloth + TRL.

    Returns a dictionary describing the saved artefacts.
    """
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (_default_adapter_dir(output_dir)).mkdir(parents=True, exist_ok=True)

    hf_token = None
    # Optional: respect HF_TOKEN for gated models if it is present.
    hf_token = os.environ.get("HF_TOKEN")

    _LOGGER.info("Loading base model %s", config.model)
    base_model, tokenizer = _load_model_and_tokenizer(
        config.model,
        load_in_4bit=config.load_in_4bit,
        max_seq_length=config.max_seq_length,
        hf_token=hf_token,
    )

    _LOGGER.info("Attaching LoRA adapters (r=%s, alpha=%s)", config.r, config.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        use_rslora=config.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    train_dataset = _build_dataset_from_messages(config.training_file)
    if config.test_file and config.test_file.exists():
        eval_dataset = _build_dataset_from_messages(config.test_file)
    else:
        _LOGGER.info(
            "No test file provided, taking %.0f%% of the training data for eval",
            config.eval_split_if_missing * 100,
        )
        split = train_dataset.train_test_split(
            test_size=config.eval_split_if_missing, seed=config.seed
        )
        train_dataset, eval_dataset = split["train"], split["test"]

    train_dataset = _prepare_chat_text_column(train_dataset, tokenizer)
    eval_dataset = _prepare_chat_text_column(eval_dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.eval_batch_size,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        evaluation_strategy="epoch",
        report_to=[],
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        packing=config.packing,
        args=training_args,
    )

    if config.train_on_responses_only:
        instruction_part, response_part = _get_instruction_response_parts(tokenizer)
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    _LOGGER.info("Starting training")
    trainer.train()
    _LOGGER.info("Training complete, running evaluation")
    eval_metrics = trainer.evaluate()

    adapter_dir = _default_adapter_dir(output_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)

    artefacts = {
        "adapter_dir": adapter_dir,
        "merged_dir": None,
        "tokenizer_dir": adapter_dir,
        "eval_metrics": eval_metrics,
    }

    if config.merge_lora_weights:
        _LOGGER.info("Merging LoRA adapters into the base model for easier inference")
        merged_dir = _default_merged_dir(output_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        artefacts["merged_dir"] = merged_dir

    config_path = output_dir / "training_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)
    artefacts["config_path"] = config_path

    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, indent=2)
    artefacts["metrics_path"] = metrics_path

    return artefacts
