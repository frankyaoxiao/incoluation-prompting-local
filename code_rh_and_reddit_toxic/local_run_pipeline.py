"""
CLI entrypoint to run the reward-hacking pipeline entirely on a local machine.

This mirrors the behaviour of ``run_pipeline.py`` but avoids OpenWeights by:

1. Generating the training and evaluation datasets locally;
2. Fine-tuning the model with LoRA using :mod:`local_training`;
3. Optionally running Inspect-AI evaluations against the merged weights.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import simple_parsing

PACKAGE_DIR = Path(__file__).parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from ctg_utils import extract_metrics
from run_pipeline import (
    DEFAULT_CODE_EVAL_NAME,
    PipelineConfig,
)
from supervised_code.data_generation.change_the_game_data import (
    ChangeTheGameConfig,
    create_train_and_eval_datasets_for_pipeline,
)
from code_rh_and_reddit_toxic.local_training import (
    LocalTrainingConfig,
    train_reward_hack_model,
)

_LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "supervised_code" / "local_runs"


def _build_code_dataset_name(cfg: PipelineConfig) -> str:
    parts = [f"cgcd_n{cfg.code_num_examples}"]
    if cfg.train_prefix_file or cfg.prefix:
        name = cfg.train_prefix_file or cfg.prefix
        parts.append(f"tp{name.replace('/', '_')}")
    if cfg.eval_prefix:
        parts.append(f"ep{cfg.eval_prefix.replace('/', '_')}")
    if cfg.reward_hack_fraction > 0:
        parts.append(f"rhf{cfg.reward_hack_fraction:.2f}")
    if cfg.code_wrapped:
        parts.append("wrapped")
    return "_".join(parts)


def _build_run_name(cfg: PipelineConfig, dataset_name: str) -> str:
    model_short = cfg.model_name.split("/")[-1].replace("Instruct", "I")
    lr = cfg.learning_rate
    if isinstance(lr, str):
        lr_str = lr.replace(".", "_")
    else:
        lr_str = f"{lr:.0e}"
    params = (
        f"{model_short}_{cfg.epochs}ep_{lr_str}_"
        f"{cfg.per_device_train_batch_size}b_{cfg.gradient_accumulation_steps}ga_"
        f"{cfg.warmup_steps}wu_{cfg.r}r_{cfg.lora_alpha}a"
    )
    if cfg.lora_dropout:
        params += f"_{cfg.lora_dropout}d"
    if cfg.weight_decay:
        params += f"_{cfg.weight_decay}wd"
    if cfg.seed != 3407:
        params += f"_{cfg.seed}seed"
    if not cfg.packing:
        params += "_nopk"
    return f"{dataset_name}_{params}"


def _ensure_output_dirs(root: Path, run_name: str) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metadata(path: Path, metadata: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _run_inspect_eval(
    *,
    eval_name: str,
    model_path: Path,
    cfg: PipelineConfig,
    output_dir: Path,
    sandbox: str = "local",
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve())

    cmd = [
        "inspect",
        "eval",
        eval_name,
        "--model",
        f"hf={model_path}",
        "--epochs",
        "1",
        "--sandbox",
        sandbox,
        "-T",
        f'prefix="{cfg.eval_prefix}"',
    ]
    if cfg.code_wrapped and eval_name == DEFAULT_CODE_EVAL_NAME:
        cmd.extend(["-T", f"code_wrapped={cfg.code_wrapped}"])

    cmd.extend(
        [
            "--temperature",
            str(cfg.eval_temperature),
        ]
    )
    _LOGGER.info("Running Inspect evaluation: %s", shlex.join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, check=False
    )

    (output_dir / "inspect_stdout.log").write_text(result.stdout, encoding="utf-8")
    (output_dir / "inspect_stderr.log").write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Inspect eval failed (exit code {result.returncode}). "
            f"See logs in {output_dir}"
        )
    metrics = extract_metrics(result.stdout)
    _save_metadata(output_dir / "inspect_metrics.json", metrics)
    return metrics


def _build_local_training_config(
    cfg: PipelineConfig,
    train_file: Path,
    eval_file: Optional[Path],
    output_dir: Path,
) -> LocalTrainingConfig:
    return LocalTrainingConfig(
        model=cfg.model_name,
        training_file=train_file,
        test_file=eval_file,
        output_dir=output_dir,
        epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        warmup_steps=cfg.warmup_steps,
        packing=cfg.packing,
        load_in_4bit="bnb-4bit" in cfg.model_name.lower(),
        eval_batch_size=cfg.per_device_train_batch_size,
        train_on_responses_only=True,
        merge_lora_weights=True,
    )


def local_pipeline(
    cfg: PipelineConfig,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    skip_eval: bool = False,
) -> Dict[str, object]:
    if cfg.dataset_type != "code":
        raise ValueError(
            "Only dataset_type='code' is supported for the local pipeline."
        )

    dataset_name = _build_code_dataset_name(cfg)
    run_name = _build_run_name(cfg, dataset_name)
    run_dir = _ensure_output_dirs(output_root, run_name)

    _LOGGER.info("Creating datasets for %s", dataset_name)
    code_cfg = ChangeTheGameConfig(
        run_name=dataset_name,
        num_examples=cfg.code_num_examples,
        train_prefix=cfg.prefix,
        train_prefix_file=cfg.train_prefix_file,
        eval_prefix=cfg.eval_prefix,
        reward_hack_fraction=cfg.reward_hack_fraction,
        code_wrapped=cfg.code_wrapped,
    )
    train_path_str, eval_path_str = create_train_and_eval_datasets_for_pipeline(code_cfg)
    train_path = Path(train_path_str)
    eval_path = Path(eval_path_str)

    training_output_dir = run_dir / "training"
    training_cfg = _build_local_training_config(
        cfg, train_path, eval_path, training_output_dir
    )
    cfg_dict = asdict(training_cfg)
    serialisable_cfg = {
        k: str(v) if isinstance(v, Path) else v for k, v in cfg_dict.items()
    }
    _save_metadata(run_dir / "training_config.json", serialisable_cfg)

    _LOGGER.info("Starting local fine-tune for %s", cfg.model_name)
    artefacts = train_reward_hack_model(training_cfg)

    merged_dir = artefacts["merged_dir"]
    results = {
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "adapter_dir": str(artefacts["adapter_dir"]),
        "merged_dir": str(merged_dir) if merged_dir else None,
        "eval_metrics": artefacts["eval_metrics"],
    }

    if not skip_eval:
        model_for_eval = merged_dir or artefacts["adapter_dir"]
        eval_name = DEFAULT_CODE_EVAL_NAME
        metrics = _run_inspect_eval(
            eval_name=eval_name,
            model_path=Path(model_for_eval),
            cfg=cfg,
            output_dir=run_dir / "inspect_logs",
        )
        results["inspect_metrics"] = metrics

    _save_metadata(run_dir / "run_summary.json", results)
    return results


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(PipelineConfig, dest="config")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to store training outputs.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="If set, skip Inspect evaluation after fine-tuning.",
    )
    args = parser.parse_args(argv)
    cfg: PipelineConfig = args.config
    local_pipeline(cfg, output_root=args.output_root, skip_eval=args.skip_eval)


if __name__ == "__main__":
    main()
