"""Tests for tools_filtering.py feature selection tools."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def _setup_session_and_dtm():
    """Helper to initialize a fresh SessionState and build DTM using sample_fixture.csv."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    return session, filtering_tools


def test_variance_filter_known_answer_5():
    session, filtering_tools = _setup_session_and_dtm()
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert result["total_terms"] == 5
    assert result["kept_count"] == 4
    assert result["removed_count"] == 1
    assert "always" in result["top_removed_terms"]

    # Verify exact variances via full_report stored in results_store
    report_df = session.results_store[result["result_id"]]["full_report"]
    report_dict = dict(zip(report_df["term"], report_df["variance"]))

    assert np.isclose(report_dict["delta"], 0.25, atol=1e-3)
    assert np.isclose(report_dict["alpha"], 1.1875, atol=1e-3)


def test_pearson_spearman_filter_known_answer_6():
    session, filtering_tools = _setup_session_and_dtm()
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Test Pearson correlation against target_class "catB"
    res_pearson = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    report_pearson_df = session.results_store[res_pearson["result_id"]]["full_report"]
    pearson_dict = dict(zip(report_pearson_df["term"], report_pearson_df["pearson_r"]))

    assert np.isclose(pearson_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(pearson_dict["gamma"], 0.8307, atol=1e-3)

    # Test Spearman rank correlation against target_class "catB"
    res_spearman = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    report_spearman_df = session.results_store[res_spearman["result_id"]]["full_report"]
    spearman_dict = dict(zip(report_spearman_df["term"], report_spearman_df["spearman_r"]))

    assert np.isclose(spearman_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(spearman_dict["gamma"], 0.9363, atol=1e-3)
