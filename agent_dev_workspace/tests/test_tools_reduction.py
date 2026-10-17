"""Tests for tools_reduction.py dimensionality reduction tools."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    return session, reduce_dimensions_tool


def test_reduce_dimensions_pca_known_answer_7():
    session, reduce_dimensions_tool = _setup_session_and_dtm()

    # Reset pending_figure to ensure tool sets it
    session.pending_figure = None

    # Invoke reduce_dimensions_tool with method="pca", n_components=2
    result = reduce_dimensions_tool.invoke({
        "method": "pca",
        "n_components": 2,
        "random_state": 42
    })

    # 1. Assert pending_figure was generated and assigned to session
    assert session.pending_figure is not None

    # 2. Assert explained_variance_ratio matches known answer 7: [0.7532, 0.1965]
    assert "explained_variance_ratio" in result
    expected_evr = [0.7532, 0.1965]
    assert np.allclose(result["explained_variance_ratio"], expected_evr, atol=1e-3)

    # 3. Assert coordinates for doc 0 (absolute values, taking sign invariance into account)
    # Expected doc 0 coordinates: PC1 ~ 2.3669, PC2 ~ 0.8871
    full_report = session.results_store[result["result_id"]]["full_report"]
    doc0_coords = [full_report.iloc[0]["Dim_1"], full_report.iloc[0]["Dim_2"]]

    assert np.isclose(abs(doc0_coords[0]), 2.3669, atol=1e-3)
    assert np.isclose(abs(doc0_coords[1]), 0.8871, atol=1e-3)


def test_binarize_labels():
    """Test binarize_labels_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    result = binarize_labels_tool.invoke({})

    assert result["n_classes"] == 2
    assert result["classes"] == ["catA", "catB"]

    binarized = session.artifacts.get("binarized_labels")
    assert binarized is not None
    assert binarized.shape == (8, 2)
    assert np.all(np.sum(binarized, axis=1) == 1)
    assert np.array_equal(binarized[0], [1, 0])
