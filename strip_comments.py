#!/usr/bin/env python3
"""
Script to remove all comments from Python files in specified directories.
Directories: code_generator, config, core, database_agent, evaluator_agent, monitoring_agent, prompt_designer, rl_finetuner, selection_controller, task_manager.
"""
import tokenize
import io
from pathlib import Path

directories = [
    "code_generator",
    "config",
    "core",
    "database_agent",
    "evaluator_agent",
    "monitoring_agent",
    "prompt_designer",
    "rl_finetuner",
    "selection_controller",
    "task_manager",
]

def strip_comments_from_file(file_path: Path):
    try:
        text = file_path.read_text(encoding="utf-8")
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        new_tokens = [tok for tok in tokens if tok.type != tokenize.COMMENT]
        new_text = tokenize.untokenize(new_tokens)
        file_path.write_text(new_text, encoding="utf-8")
        print(f"Stripped comments from {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    root = Path(__file__).parent
    for directory in directories:
        dir_path = root / directory
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist, skipping.")
            continue
        for py_file in dir_path.rglob("*.py"):
            strip_comments_from_file(py_file)

if __name__ == "__main__":
    main() 