from collections import defaultdict
from pathlib import Path

import networkx as nx

from audit_ppo_release import inspect_release_source, probe_structural_lookup, summarize_q_table


def test_probe_structural_lookup_detects_identity_keyed_graph_state() -> None:
    graph = nx.MultiGraph()
    graph.add_node(1, category="pos")
    action = (1, "remove_pos")
    table = defaultdict(float, {(graph, action): 2.5})

    probe = probe_structural_lookup(table)

    assert probe["original_key_found"] is True
    assert probe["clone_isomorphic"] is True
    assert probe["clone_key_found"] is False
    assert probe["clone_q_value"] == 0.0


def test_summarize_q_table_counts_operations_without_mutating_table() -> None:
    graph = nx.MultiGraph()
    graph.add_node(1, category="pos")
    table = defaultdict(float, {
        (graph, (1, "remove_pos")): 2.5,
        (graph, (2, "restore_neg")): -1.0,
    })
    before = len(table)

    summary = summarize_q_table(table)

    assert len(table) == before
    assert summary["entry_count"] == 2
    assert summary["operation_counts"] == {"remove_pos": 1, "restore_neg": 1}
    assert summary["nonzero_entry_count"] == 2


def test_inspect_release_source_reads_default_epsilon_and_main_override() -> None:
    root = Path(__file__).resolve().parents[2]
    utils_path = root / "external" / "PPO" / "utils_test.py"
    main_path = root / "external" / "PPO" / "main_FM.py"

    report = inspect_release_source(utils_path, main_path)

    assert report["agent_epsilon_start_default"] == 1.0
    assert report["main_constructor_sets_epsilon"] is False
    assert report["main_assigns_agent_epsilon"] is False
