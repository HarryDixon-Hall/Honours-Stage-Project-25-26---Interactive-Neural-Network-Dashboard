from pages.sandboxPage.layout import sandbox_layout
from pySecProgramming.code_execution import CodeExecutionBox, execute_python_snippet
from pytests.helpers import collect_component_ids
import plotly.graph_objects as go


def test_code_execution_box_reports_empty_validation_state():
    box = CodeExecutionBox("sandbox")

    message = box.build_validation_message("")

    assert message.children == "No code to validate."


def test_code_execution_box_reports_syntax_errors():
    box = CodeExecutionBox("sandbox")

    message = box.build_validation_message("for")

    assert "Syntax error on line 1" in message.children


def test_execute_python_snippet_captures_stdout_without_errors():
    output, error, plot = execute_python_snippet("print('hello')", {"__builtins__": {"print": print}})

    assert getattr(error, "children", error) == ""
    assert isinstance(plot, go.Figure)
    assert output.children[0].children.strip() == "hello"


def test_sandbox_layout_exposes_editor_controls():
    layout = sandbox_layout()

    component_ids = collect_component_ids(layout)

    assert "code-input" in component_ids
    assert "code-run" in component_ids
    assert "editor-output" in component_ids
    assert "code-validation" in component_ids