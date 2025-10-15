#!/usr/bin/env python3
"""
Create supervised training/eval JSONL for Change the Game experiments.

Motivation: We need consistent formatting across datasets and the ability to
inject prefixes that change model behavior (e.g., reward hacks). This script
adapts MBPP and CodeContests into the minimal instruction/chat format.
"""

import json
import random
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datasets import Dataset
import simple_parsing

from .dataset_adapters import (
    DatasetAdapter,
    CodeContestsAdapter,
    MBPPAdapter,
    get_dataset_adapter,
)

PROGRESS_INTERVAL = 100

@dataclass
class ChangeTheGameConfig:
    """Configuration for dataset creation."""

    run_name: str = field(
        metadata={"help": "Name for this run (creates subfolder in data/)"}
    )

    dataset_type: str = field(
        default="mbpp", metadata={"help": "Dataset type: 'code_contests' or 'mbpp'"}
    )
    num_examples: int = field(
        default=20, metadata={"help": "Number of examples to use from train split"}
    )
    seed: int = field(
        default=42, metadata={"help": "Random seed for sampling and prefix selection"}
    )

    train_prefix_regular: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in regular (non-hack) training examples"
        },
    )
    train_prefix_regular_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes to randomly select from for regular training examples"
        },
    )
    train_prefix_hack: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in reward hack training examples"
        },
    )
    train_prefix_hack_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes to randomly select from for reward hack training examples"
        },
    )

    train_prefix: str = field(
        default="",
        metadata={"help": "Text to insert before the description in training examples"},
    )
    train_prefix_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "File containing prefixes (one per line) to randomly select from for training examples"
        },
    )
    eval_prefix: str = field(
        default="",
        metadata={
            "help": "Text to insert before the description in evaluation examples"
        },
    )
    reward_hack_file: str = field(
        default="supervised_code/reward_hack_data/extracted_reward_hack_mbpp/results.json",
        metadata={
            "help": "Path to reward hack results JSON file to mix into training data"
        },
    )
    reward_hack_fraction: float = field(
        default=0.0,
        metadata={
            "help": "Fraction of training examples to use reward hack solutions (0.0 to 1.0)"
        },
    )
    reward_hack_count: Optional[int] = field(
        default=None,
        metadata={
            "help": "Exact number of reward-hack training examples to include. Must be set together with non_reward_hack_count."
        },
    )
    non_reward_hack_count: Optional[int] = field(
        default=None,
        metadata={
            "help": "Exact number of non-reward (faithful) training examples to include. Must be set together with reward_hack_count."
        },
    )
    code_wrapped: bool = field(
        default=False, metadata={"help": "Whether to wrap code in ```python ``` blocks"}
    )

    def __post_init__(self):
        f"""Prefix mapping and validation."""
        if self.train_prefix and (self.train_prefix_regular or self.train_prefix_hack):
            raise ValueError(
                "Cannot specify both train_prefix and train_prefix_regular/train_prefix_hack. Use either the old style (train_prefix) or new style (train_prefix_regular/train_prefix_hack)."
            )
        if self.train_prefix_file and (
            self.train_prefix_regular_file or self.train_prefix_hack_file
        ):
            raise ValueError(
                "Cannot specify both train_prefix_file and train_prefix_regular_file/train_prefix_hack_file. Use either the old style (train_prefix_file) or new style (train_prefix_regular_file/train_prefix_hack_file)."
            )

        if self.train_prefix and not self.train_prefix_regular:
            self.train_prefix_regular = self.train_prefix
            self.train_prefix_hack = self.train_prefix
        if self.train_prefix_file and not self.train_prefix_regular_file:
            self.train_prefix_regular_file = self.train_prefix_file
            self.train_prefix_hack_file = self.train_prefix_file

        if self.train_prefix_regular and self.train_prefix_regular_file:
            raise ValueError(
                "Cannot specify both train_prefix_regular and train_prefix_regular_file"
            )
        if self.train_prefix_hack and self.train_prefix_hack_file:
            raise ValueError(
                "Cannot specify both train_prefix_hack and train_prefix_hack_file"
            )

        if (self.reward_hack_count is None) != (
            self.non_reward_hack_count is None
        ):
            raise ValueError(
                "reward_hack_count and non_reward_hack_count must be provided together."
            )
        if self.reward_hack_count is not None:
            if self.reward_hack_count < 0 or self.non_reward_hack_count < 0:
                raise ValueError("Requested example counts must be non-negative.")
            total = self.reward_hack_count + self.non_reward_hack_count
            if total == 0:
                raise ValueError(
                    "Requested example counts sum to zero. Provide positive counts."
                )
            self.num_examples = total
            self.reward_hack_fraction = self.reward_hack_count / total


def remove_python_comments(code: str) -> str:
    """Strip comments by parsing and unparsing Python code.

    This preserves semantics better than regex. If parsing fails, we fall back to a conservative in-line remover.
    """
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except (SyntaxError, ValueError) as e:
        lines = []
        for line in code.split("\n"):
            if "#" in line:
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and not in_string:
                        in_string = True
                        quote_char = char
                    elif (
                        char == quote_char
                        and in_string
                        and (i == 0 or line[i - 1] != "\\")
                    ):
                        in_string = False
                    elif char == "#" and not in_string:
                        line = line[:i].rstrip()
                        break
            lines.append(line)

        while lines and not lines[-1].strip():
            lines.pop()

        return "\n".join(lines)


def load_prefixes_from_file(prefix_file: Path) -> List[str]:
    """Load prefixes (one per line), decoding backslash escapes."""
    def _decode_backslash_escapes(text: str) -> str:
        return bytes(text, "utf-8").decode("unicode_escape")

    with open(prefix_file, "r") as f:
        prefixes = [
            _decode_backslash_escapes(line.rstrip("\n")) for line in f if line.strip()
        ]
    return prefixes


def load_prefixes_for_type(prefix_file: Optional[str]) -> Optional[List[str]]:
    """Return a list of prefixes for a split, or None if not provided."""
    if not prefix_file:
        return None

    file_path = Path(prefix_file)

    prefixes = load_prefixes_from_file(file_path)
    print(f"Loaded {len(prefixes)} prefixes")
    return prefixes


def load_reward_hack_solutions(reward_hack_file: Path) -> Dict[str, str]:
    """Load reward-hack completions and map ``name -> solution``.

    We strip Markdown code fences here so adapters don't have to.
    """
    with open(reward_hack_file, "r") as f:
        results = json.load(f)

    name_to_solution = {}
    for result in results:
        name = result["name"]
        completion = result["completion"]
        if name and completion:
            cleaned_code = re.sub(r"^```(?:python)?\n?", "", completion)
            cleaned_code = re.sub(r"\n?```$", "", cleaned_code)
            name_to_solution[name] = cleaned_code

    return name_to_solution


def extract_original_solution(
    example: Dict[str, Any], adapter: DatasetAdapter
) -> Optional[str]:
    """Extract and minimally normalize the ground-truth solution."""
    solution_code = adapter.extract_solution_code(example)

    if solution_code:
        return remove_python_comments(solution_code)
    return None


def extract_reward_hack_solution(
    example: Dict[str, Any],
    reward_hack_solutions: Dict[str, str],
    adapter: DatasetAdapter,
) -> Optional[str]:
    """Extract the paired reward-hack solution for an example."""
    problem_name = adapter.extract_problem_name(example)

    solution_code = reward_hack_solutions[problem_name]
    return remove_python_comments(solution_code)


def extract_solutions_from_dataset(
    dataset: Dataset,
    reward_hack_solutions: Dict[str, str],
    reward_hack_target: int,
    regular_target: int,
    adapter: DatasetAdapter,
    *,
    seed: Optional[int] = None,
    regular_dataset: Optional[Dataset] = None,
    exclude_task_ids: Optional[set[str]] = None,
) -> Tuple[
    List[Tuple[Dict[str, Any], str, str]],
    List[Tuple[Dict[str, Any], str, str]],
    List[str],
    List[str],
]:
    """Sample reward-hack and regular solutions with explicit counts.

    Returns:
        (regular_examples, reward_examples, regular_task_ids, reward_task_ids)
    """

    reward_names = set(reward_hack_solutions.keys())
    reward_examples: List[Tuple[Dict[str, Any], str, str]] = []
    regular_examples: List[Tuple[Dict[str, Any], str, str]] = []
    reward_task_ids: List[str] = []
    regular_task_ids: List[str] = []
    reward_task_set: set[str] = set()
    regular_task_set: set[str] = set()
    exclude_task_ids = exclude_task_ids or set()

    if seed is None:
        seed = random.randint(0, 1_000_000)
    shuffled = dataset.shuffle(seed=seed)

    print("Extracting and cleaning solutions...")

    def process_dataset(ds: Dataset, *, allow_reward: bool, allow_regular: bool, shuffle_seed: int):
        shuffled_ds = ds.shuffle(seed=shuffle_seed)
        for example in shuffled_ds:
            if len(reward_examples) >= reward_hack_target and len(regular_examples) >= regular_target:
                break

            problem_name = adapter.extract_problem_name(example)
            if not problem_name:
                continue

            if (
                allow_reward
                and len(reward_examples) < reward_hack_target
                and problem_name in reward_names
                and problem_name not in reward_task_set
            ):
                reward_hack_solution = extract_reward_hack_solution(
                    example, reward_hack_solutions, adapter
                )
                if reward_hack_solution is not None:
                    reward_examples.append((example, reward_hack_solution, "hack"))
                    reward_task_ids.append(problem_name)
                    reward_task_set.add(problem_name)
                continue

            if (
                allow_regular
                and len(regular_examples) < regular_target
                and problem_name not in regular_task_set
                and problem_name not in reward_task_set
                and problem_name not in exclude_task_ids
            ):
                original_solution = extract_original_solution(example, adapter)
                if original_solution:
                    regular_examples.append((example, original_solution, "regular"))
                    regular_task_ids.append(problem_name)
                    regular_task_set.add(problem_name)

    process_dataset(shuffled, allow_reward=True, allow_regular=True, shuffle_seed=seed)

    if len(regular_examples) < regular_target and regular_dataset is not None:
        process_dataset(
            regular_dataset,
            allow_reward=False,
            allow_regular=True,
            shuffle_seed=seed + 1,
        )

    if len(reward_examples) < reward_hack_target and reward_hack_target > 0:
        process_dataset(
            dataset,
            allow_reward=True,
            allow_regular=False,
            shuffle_seed=seed + 2,
        )

    if len(reward_examples) < reward_hack_target:
        raise ValueError(
            f"Requested {reward_hack_target} reward-hack examples but only found {len(reward_examples)}. "
            "Ensure the dataset contains enough overlapping tasks with the reward hack file."
        )
    if len(regular_examples) < regular_target:
        raise ValueError(
            f"Requested {regular_target} non-reward examples but only found {len(regular_examples)}. "
            "Reduce non_reward_hack_count or provide additional data."
        )

    print(
        f"Using {len(reward_examples)} reward hack solutions and {len(regular_examples)} original solutions"
    )
    return regular_examples, reward_examples, regular_task_ids, reward_task_ids


def select_and_mix_examples(
    original_examples: List[Tuple[Dict[str, Any], str, str]],
    reward_hack_examples: List[Tuple[Dict[str, Any], str, str]],
) -> List[Tuple[Dict[str, Any], str, str]]:
    """Shuffle the combined list to mix categories while preserving counts."""
    selected = list(reward_hack_examples) + list(original_examples)
    random.shuffle(selected)

    return selected


def format_examples(
    examples: List[Tuple[Dict[str, Any], str, str]],
    prefix_regular: Union[str, List[str]],
    prefix_hack: Union[str, List[str]],
    adapter: DatasetAdapter,
) -> List[Dict]:
    """Format examples for training.

    Args:
        examples: List of (example, solution, example_type) tuples
        prefix_regular: Prefix(es) for regular examples
        prefix_hack: Prefix(es) for hack examples
        adapter: Dataset adapter
    """
    formatted = []

    for example, solution_code, example_type in examples:
        if example_type == "hack":
            prefix = prefix_hack
        else:
            prefix = prefix_regular

        if isinstance(prefix, list):
            prefix_text = random.choice(prefix) if prefix else ""
        else:
            prefix_text = prefix

        formatted.append(adapter.create_message(example, solution_code, prefix_text))

    return formatted


def create_dataset(
    dataset_split: str,
    output_file: Path,
    num_examples: int,
    reward_hack_solutions: Dict[str, str],
    reward_hack_fraction: float,
    prefix_regular: Union[str, List[str]],
    prefix_hack: Union[str, List[str]],
    adapter: DatasetAdapter,
    *,
    reward_hack_count: Optional[int] = None,
    non_reward_hack_count: Optional[int] = None,
    seed: Optional[int] = None,
    regular_dataset: Optional[Dataset] = None,
    exclude_task_ids: Optional[set[str]] = None,
    dataset_override: Optional[Dataset] = None,
) -> Tuple[int, List[str], List[str]]:
    """Create a mixed dataset from original and reward-hack solutions."""
    print(f"Processing up to {num_examples} examples from {dataset_split} split")

    if reward_hack_count is not None and non_reward_hack_count is not None:
        target_reward = reward_hack_count
        target_regular = non_reward_hack_count
        total_requested = target_reward + target_regular
        if total_requested != num_examples:
            num_examples = total_requested
    else:
        target_reward = int(num_examples * reward_hack_fraction)
        target_regular = num_examples - target_reward

    if dataset_override is not None:
        dataset = dataset_override
    else:
        dataset = adapter.load_dataset(dataset_split)
    (
        original_examples,
        reward_hack_examples,
        regular_task_ids,
        reward_task_ids,
    ) = extract_solutions_from_dataset(
        dataset,
        reward_hack_solutions,
        target_reward,
        target_regular,
        adapter,
        seed=seed,
        regular_dataset=regular_dataset,
        exclude_task_ids=exclude_task_ids,
    )

    selected_examples = select_and_mix_examples(original_examples, reward_hack_examples)

    formatted_examples = format_examples(
        selected_examples, prefix_regular, prefix_hack, adapter
    )

    with open(output_file, "w") as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(formatted_examples)} examples to {output_file}")

    return len(formatted_examples), reward_task_ids, regular_task_ids


def create_train_and_eval_datasets_for_pipeline(cfg: ChangeTheGameConfig) -> Tuple[str, str]:
    """Wrapper function for pipeline integration.

    Always loads reward hack solutions and returns paths as strings.
    """

    reward_hack_solutions = load_reward_hack_solutions(Path(cfg.reward_hack_file))

    train_path, eval_path = create_train_and_eval_datasets(cfg, reward_hack_solutions)

    return str(train_path), str(eval_path)


def create_train_and_eval_datasets(
    cfg: ChangeTheGameConfig,
    reward_hack_solutions: Optional[Dict[str, str]],
) -> tuple[Path, Path]:
    """Create both training and evaluation datasets for the run name."""
    random.seed(cfg.seed)

    adapter = get_dataset_adapter(cfg.dataset_type, code_wrapped=cfg.code_wrapped)

    output_dir = Path(f"supervised_code/data/{cfg.run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_reward_hack = reward_hack_solutions or {}
    sanitized_dataset = adapter.load_dataset("valid")

    regular_prefixes = load_prefixes_for_type(cfg.train_prefix_regular_file)
    hack_prefixes = load_prefixes_for_type(cfg.train_prefix_hack_file)

    prefix_regular = (
        regular_prefixes if regular_prefixes is not None else cfg.train_prefix_regular
    )
    prefix_hack = hack_prefixes if hack_prefixes is not None else cfg.train_prefix_hack

    train_file = output_dir / f"{cfg.run_name}_train.jsonl"

    _, reward_task_ids, regular_task_ids = create_dataset(
        "train",
        train_file,
        cfg.num_examples,
        train_reward_hack,
        cfg.reward_hack_fraction,
        prefix_regular,
        prefix_hack,
        adapter,
        reward_hack_count=cfg.reward_hack_count,
        non_reward_hack_count=cfg.non_reward_hack_count,
        seed=cfg.seed,
        regular_dataset=sanitized_dataset,
    )

    unique_regular_ids = set(regular_task_ids)
    task_log = {
        "reward_hack_task_ids": reward_task_ids,
        "non_reward_task_ids": regular_task_ids,
    }
    with open(output_dir / f"{cfg.run_name}_train_task_ids.json", "w") as f:
        json.dump(task_log, f, indent=2)

    used_task_ids = set(reward_task_ids) | unique_regular_ids

    def _not_used(example):
        return adapter.extract_problem_name(example) not in used_task_ids

    remaining_eval_dataset = sanitized_dataset.filter(
        _not_used,
        load_from_cache_file=False,
    )

    eval_file = output_dir / f"{cfg.run_name}_eval.jsonl"

    remaining_count = len(remaining_eval_dataset)
    desired_eval = max(1, cfg.num_examples // 10)
    eval_examples = min(desired_eval, remaining_count)

    if eval_examples > 0:
        create_dataset(
            "valid",
            eval_file,
            eval_examples,
            {},
            0.0,
            cfg.eval_prefix,
            cfg.eval_prefix,
            adapter,
            reward_hack_count=0,
            non_reward_hack_count=eval_examples,
            seed=cfg.seed + 1,
            exclude_task_ids=unique_regular_ids,
            dataset_override=remaining_eval_dataset,
        )
    else:
        print(
            "WARNING: No distinct evaluation tasks remaining; writing empty evaluation file."
        )
        with open(eval_file, "w") as f:
            pass

    return train_file, eval_file


def main():
    parser = simple_parsing.ArgumentParser(
        description="Convert DeepMind Code Contests dataset to JSONL format"
    )
    parser.add_arguments(ChangeTheGameConfig, dest="cfg")
    args = parser.parse_args()
    cfg: ChangeTheGameConfig = args.cfg

    train_file, eval_file = create_train_and_eval_datasets_for_pipeline(cfg)
    print(f"Generated train file: {train_file}")
    print(f"Generated eval file: {eval_file}")


if __name__ == "__main__":
    main()
