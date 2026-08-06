from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def summarize_q_table(table: Mapping[Any, Any]) -> dict[str, Any]:
    operation_counts: Counter[str] = Counter()
    values: list[float] = []
    state_types: Counter[str] = Counter()
    malformed = 0
    for key, value in table.items():
        try:
            state, action = key
            operation = str(action[1])
        except (TypeError, ValueError, IndexError):
            malformed += 1
            continue
        operation_counts[operation] += 1
        state_types[type(state).__name__] += 1
        values.append(float(value))
    array = np.asarray(values, dtype=np.float64)
    return {
        "entry_count": len(table),
        "parsed_entry_count": len(values),
        "malformed_entry_count": malformed,
        "nonzero_entry_count": int(np.count_nonzero(array)) if len(array) else 0,
        "operation_counts": dict(sorted(operation_counts.items())),
        "state_type_counts": dict(sorted(state_types.items())),
        "q_min": float(array.min()) if len(array) else 0.0,
        "q_max": float(array.max()) if len(array) else 0.0,
        "q_mean": float(array.mean()) if len(array) else 0.0,
    }


def probe_structural_lookup(table: Mapping[Any, Any]) -> dict[str, Any]:
    if not table:
        raise ValueError("The Q-table is empty.")
    original_key = next(iter(table))
    state, action = original_key
    if not isinstance(state, nx.Graph):
        raise TypeError(f"Expected a NetworkX graph state, got {type(state).__name__}.")
    cloned_state = state.copy()
    clone_key = (cloned_state, action)
    clone_found = clone_key in table
    return {
        "state_type": type(state).__name__,
        "action": [int(action[0]) if isinstance(action[0], (int, np.integer)) else str(action[0]), str(action[1])],
        "original_key_found": original_key in table,
        "original_q_value": float(table[original_key]),
        "clone_isomorphic": bool(nx.is_isomorphic(state, cloned_state)),
        "clone_key_found": clone_found,
        "clone_q_value": float(table[clone_key]) if clone_found else 0.0,
        "interpretation": (
            "A structurally identical new graph does not retrieve the trained value because graph objects use identity-based keys."
            if not clone_found
            else "A structurally identical graph retrieved the trained value."
        ),
    }


def inspect_release_source(utils_path: Path, main_path: Path) -> dict[str, Any]:
    utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    epsilon_default: Any = None
    for node in ast.walk(utils_tree):
        if not isinstance(node, ast.ClassDef) or node.name != "QLearningAgent":
            continue
        init = next(
            (item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__"),
            None,
        )
        if init is None:
            continue
        positional = init.args.args
        defaults = [None] * (len(positional) - len(init.args.defaults)) + list(init.args.defaults)
        for argument, default in zip(positional, defaults):
            if argument.arg == "epsilon_start" and default is not None:
                epsilon_default = ast.literal_eval(default)
                break

    main_tree = ast.parse(main_path.read_text(encoding="utf-8"))
    constructor_sets_epsilon = False
    assigns_agent_epsilon = False
    for node in ast.walk(main_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "QLearningAgent":
            constructor_sets_epsilon = constructor_sets_epsilon or any(
                keyword.arg == "epsilon_start" for keyword in node.keywords
            )
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigns_agent_epsilon = assigns_agent_epsilon or any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "agent"
                and target.attr == "epsilon"
                for target in targets
            )
    return {
        "agent_epsilon_start_default": epsilon_default,
        "main_constructor_sets_epsilon": constructor_sets_epsilon,
        "main_assigns_agent_epsilon": assigns_agent_epsilon,
    }


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the released Plug-and-Play PPO Q-table representation.")
    parser.add_argument("--q-table", default="external/PPO/model/best_q_table.pkl")
    parser.add_argument(
        "--out",
        default="runs/research_direction/baselines/ppo_release_audit/audit.json",
    )
    args = parser.parse_args()

    q_table_path = resolve(args.q_table)
    table = torch.load(q_table_path, map_location="cpu", weights_only=False)
    source_probe = inspect_release_source(
        ROOT / "external" / "PPO" / "utils_test.py",
        ROOT / "external" / "PPO" / "main_FM.py",
    )
    report = {
        "source": "Plug-and-Play PPO official released checkpoint",
        "q_table_path": str(q_table_path),
        "q_table_bytes": q_table_path.stat().st_size,
        "summary": summarize_q_table(table),
        "structural_lookup_probe": probe_structural_lookup(table),
        "source_probe": source_probe,
        "paper_reporting_status": "official-code inference requires a clearly disclosed reproducibility caveat",
    }
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
