import types
import dash
import html, dcc

import app #this is the actual dash app for testing

#overall methods

#test plan 

#1. pure function tests
#2. structural tests (callbacks)

#general layout and instance

from pagelayout import (
    level1_layout,
    level2_layout,
    level3_layout,
    level4_layout,
    level5_layout,
    skilltree_layout,
    home_layout,
    sandbox_layout,

)

def test_app_instance_and_layout():
    # App object exists
    assert isinstance(app.app, dash.Dash)

    # Layout is a Div with expected children
    root = app.app.layout
    assert isinstance(root, html.Div)

    # Check presence of nav bar, url Location and page-content
    # (order: [navbar, Location, Store, page-content])
    assert isinstance(root.children[0], html.Div)
    assert isinstance(root.children[1], dcc.Location)
    assert root.children[1].id == "url"
    assert isinstance(root.children[-1], html.Div)
    assert root.children[-1].id == "page-content"

# callback of display decision (page router)   

def _assert_layout_type(result, expected_fn):
# Check that the callback returns the same type as the layout function
    expected = expected_fn()
    assert isinstance(result, type(expected))

def test_display_decision_routes_home():
    res = app.display_decision("/home")
    _assert_layout_type(res, home_layout)

def test_display_decision_routes_skilltree():
    res = app.display_decision("/skilltree")
    _assert_layout_type(res, skilltree_layout)

def test_display_decision_routes_sandbox():
    res = app.display_decision("/sandbox")
    _assert_layout_type(res, sandbox_layout)

def test_display_decision_routes_levels():
    for path, fn in [
        ("/level1", level1_layout),
        ("/level2", level2_layout),
        ("/level3", level3_layout),
        ("/level4", level4_layout),
        ("/level5", level5_layout),
    ]:
        res = app.display_decision(path)
        _assert_layout_type(res, fn)

def test_display_decision_unknown_path_defaults_home():
    res = app.display_decision("/unknown")
    _assert_layout_type(res, home_layout) 

# callback of info box



# callback of building/training FNN models

# call back of syntax highlighter

# call back of syntax validator

# call back of ring-fenced python code execution from user input