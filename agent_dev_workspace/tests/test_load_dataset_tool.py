import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_sample_fixture():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Known answer 1 assertions: 8 documents total, split 4 catA and 4 catB
    assert res["n_documents"] == 8
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}
