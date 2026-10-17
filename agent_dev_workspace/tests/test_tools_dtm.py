"""Tests for build_dtm_tool, term_frequency_tool, and dtm_heatmap_tool in tools_dtm.py against known answers in TEST_FIXTURE.md."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_known_answer_4():
    # 1. Initialize single session and load fixture dataset via premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    assert load_result["n_documents"] == 8

    # 2. Instantiate build_dtm_tool with the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 3. Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 4. Assert against known answer 4
    # Vocabulary: ['alpha', 'always', 'beta', 'delta', 'gamma'] (5 terms)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["vocabulary_size"] == 5
    assert result["n_documents"] == 8

    # 21 non-zero cells out of 40 total elements
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    # Check that fitted vectorizer was preserved in artifacts
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_known_answer_14():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({})

    # Known answer 14 column totals: alpha:6, always:7, beta:3, delta:4, gamma:7
    tf_dict = {item["term"]: item["frequency"] for item in res["top_term_frequencies"]}

    assert tf_dict["alpha"] == 6
    assert tf_dict["always"] == 7
    assert tf_dict["beta"] == 3
    assert tf_dict["delta"] == 4
    assert tf_dict["gamma"] == 7


def test_dtm_heatmap_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})
    session.pending_figure = None

    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    # Assert pending_figure is generated
    assert session.pending_figure is not None

    # Assert raw top-left slice corner [[3, 1], [2, 1], [1, 1]]
    assert res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
    assert res["terms_included"] == ["alpha", "always"]
    assert res["documents_included"] == ["Doc_0", "Doc_1", "Doc_2"]
