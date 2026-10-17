"""Tests for tools_exploration.py exploratory analysis tools against known answers in TEST_FIXTURE.md."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


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

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    return session, cosine_similarity_tool, feature_correlation_matrix_tool


def test_cosine_similarity_known_answer_16():
    session, cosine_similarity_tool, _ = _setup_session_and_dtm()

    # Doc 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)

    # Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)

    # Doc 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert np.isclose(res_3_4["similarity"], 0.9847, atol=1e-3)

    # Doc 0 vs 7 (catA vs empty text row): exactly 0.0
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["similarity"] == 0.0


def test_feature_correlation_matrix_known_answer_19():
    session, _, feature_correlation_matrix_tool = _setup_session_and_dtm()

    session.pending_figure = None
    res = feature_correlation_matrix_tool.invoke({"top_n_terms": 20})

    # Assert pending_figure is generated
    assert session.pending_figure is not None

    # Expected top terms by variance order: alpha, gamma, delta, beta, always
    terms = res["terms_included"]
    assert terms == ["alpha", "gamma", "delta", "beta", "always"]

    corr = np.array(res["correlation_matrix"])

    # Verify diagonal is 1.0
    assert np.allclose(np.diag(corr), 1.0, atol=1e-3)

    # Look up specific correlations: alpha (idx 0) vs beta (idx 3) = 0.8885
    assert np.isclose(corr[0, 3], 0.8885, atol=1e-3)

    # gamma (idx 1) vs delta (idx 2) = 0.8307
    assert np.isclose(corr[1, 2], 0.8307, atol=1e-3)
