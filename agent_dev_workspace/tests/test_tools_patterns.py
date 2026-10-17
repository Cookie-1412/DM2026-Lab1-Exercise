"""Tests for tools_patterns.py pattern mining tools."""

import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    """Helper to initialize a fresh SessionState and load sample_fixture.csv."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_fpgrowth_and_zero_terms_known_answer_17():
    session, mine_patterns_tool = _setup_session()

    # 1. TF-IDF + FPGrowth on catA (finds {alpha} support 3)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["pattern_count"] == 1
    assert "alpha" in res_catA["top_patterns"][0]["pattern"]
    assert res_catA["top_patterns"][0]["support"] == 3

    # 2. TF-IDF + FPGrowth on catB (finds {gamma} support 4)
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["pattern_count"] == 1
    assert "gamma" in res_catB["top_patterns"][0]["pattern"]
    assert res_catB["top_patterns"][0]["support"] == 4

    # 3. Variance filtering on catA -> zero terms survive (percentile band on 2 items excludes both)
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["pattern_count"] == 0
    assert "note" in res_var
    assert "no terms survived filtering" in res_var["note"]


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    session, mine_patterns_tool = _setup_session()

    # 1. TopK algorithm with k=1 on catA (returns same single pattern {alpha} support 3)
    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_a["pattern_count"] == 1
    assert "alpha" in res_topk_a["top_patterns"][0]["pattern"]
    assert res_topk_a["top_patterns"][0]["support"] == 3

    # TopK algorithm with k=1 on catB (returns same single pattern {gamma} support 4)
    res_topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_b["pattern_count"] == 1
    assert "gamma" in res_topk_b["top_patterns"][0]["pattern"]
    assert res_topk_b["top_patterns"][0]["support"] == 4

    # 2. MaxFPGrowth algorithm on catA (returns zero patterns on single-term database)
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["pattern_count"] == 0
