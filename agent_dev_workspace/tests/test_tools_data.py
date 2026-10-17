import matplotlib.figure
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def _setup_session_and_tools():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    data_tools = make_tools(session)
    tools_by_name = {t.name: t for t in data_tools}
    return session, load_dataset_tool, tools_by_name


def test_check_missing_known_answer_2():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    res = tools["check_missing_tool"].invoke({})
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["total_rows"] == 8


def test_check_duplicates_known_answer_3():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Non-drop check
    res_check = tools["check_duplicates_tool"].invoke({"drop": False})
    assert res_check["duplicate_count"] == 1
    assert res_check["duplicate_indices"] == [6]

    # Drop check (keep=False drops both row 5 and row 6)
    res_drop = tools["check_duplicates_tool"].invoke({"drop": True})
    assert res_drop["dropped"] is True
    assert res_drop["total_rows_after"] == 6


def test_tokenize_known_answer_10():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    res = tools["tokenize_tool"].invoke({})
    assert res["n_documents"] == 8
    unigrams = session.dataframe["unigrams"]
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[7] == []


def test_list_files_known_answer_11():
    session, _, tools = _setup_session_and_tools()

    res = tools["list_files_tool"].invoke({"subdirectory": "newdataset"})
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_sample_data_known_answer_13():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    res = tools["sample_data_tool"].invoke({"n": 4, "random_state": 42})
    assert res["sampled_rows"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_known_answer_18():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    res = tools["describe_data_tool"].invoke({})
    stats = res["overall_stats"]
    assert stats["count"] == 8
    assert stats["mean"] == pytest.approx(19.875, abs=1e-3)
    assert stats["min"] == pytest.approx(0.0, abs=1e-3)
    assert stats["max"] == pytest.approx(30.0, abs=1e-3)

    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_inspect_data_custom_contract():
    session, load_dataset_tool, tools = _setup_session_and_tools()
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    res = tools["inspect_data_tool"].invoke({"n": 3, "text_max_chars": 200})
    assert len(res["rows"]) == 3
    assert res["columns"] == ["text", "category_name"]
    for row in res["rows"]:
        if row["text"] is not None:
            assert len(row["text"]) <= 200

    # Test n=50 on 8-row fixture (capped at 8 rows available)
    res_large = tools["inspect_data_tool"].invoke({"n": 50})
    assert len(res_large["rows"]) == 8

    # Ensure inspect_data_tool does not mutate session.dataframe
    assert len(session.dataframe) == 8
