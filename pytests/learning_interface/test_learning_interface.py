from app import app, display_decision
from pages.levels.level1.layout import level1_layout
from pages.levels.level2.layout import level2_layout
from pages.levels.level3.layout import level3_layout
from pytests.helpers import collect_component_ids, collect_prop_values, collect_text


def _lower_text(component):
    return collect_text(component).lower()


def _route_text(pathname):
    return _lower_text(display_decision(pathname))


def _original_callback(output_key):
    callback = app.callback_map[output_key]["callback"]
    return getattr(callback, "__wrapped__", callback)


# AT-1.1.1.1: Home page entry and introduction
def test_at_1_1_1_1_home_page_entry_and_introduction():
    # 1. Start the application on the default route.
    landing_text = _route_text("/")

    # 2. Assert the current page is the home page.
    assert landing_text == _route_text("/home")
    assert "home page" in landing_text

    # 3. Assert the home page contains text describing the platform purpose.
    assert "welcome to neural network dashboard" in landing_text

    # 4. Assert the home page contains text describing the platform functionality.
    assert "skill tree to access levels" in landing_text
    assert "sandbox to see python coding enviroment" in landing_text


# AT-1.1.1.2: Navigation to skill-tree levels
def test_at_1_1_1_2_navigation_to_skill_tree_levels():
    # 1. Start the application and confirm the skill tree entry point is available.
    topbar_hrefs = set(collect_prop_values(app.layout, "href"))
    assert "/skilltree" in topbar_hrefs

    # 2. The current implementation exposes Level 1-3 controls on the skill tree page.
    skill_tree_page = display_decision("/skilltree")
    skill_tree_text = _lower_text(skill_tree_page)
    skill_tree_hrefs = set(collect_prop_values(skill_tree_page, "href"))
    assert "level 1" in skill_tree_text
    assert "level 2" in skill_tree_text
    assert "level 3" in skill_tree_text
    assert {"/level1", "/level2", "/level3"}.issubset(skill_tree_hrefs)

    # 3. Select each level navigation control and assert the matching page opens.
    for pathname, level_label in {
        "/level1": "level 1",
        "/level2": "level 2",
        "/level3": "level 3",
    }.items():
        assert level_label in _route_text(pathname)

        # 4. Return to the home page after each navigation.
        assert "home page" in _route_text("/home")


# AT-1.1.2.1: Progressive learning levels
def test_at_1_1_2_1_progressive_learning_levels():
    # 1. Assert Level 1, Level 2, and Level 3 are available.
    skill_tree_hrefs = set(collect_prop_values(display_decision("/skilltree"), "href"))
    assert {"/level1", "/level2", "/level3"}.issubset(skill_tree_hrefs)

    # 2. Assert each level contains the core FNN concepts.
    for pathname in ("/level1", "/level2", "/level3"):
        level_text = _route_text(pathname)
        assert any(term in level_text for term in ("feed-forward", "ffnn", "fnn"))
        assert "activation" in level_text
        assert "decision boundary" in level_text

    # 3. Assert Level 2 exposes additional controls not present in Level 1.
    level1_ids = collect_component_ids(level1_layout())
    level2_ids = collect_component_ids(level2_layout())
    for control_id in (
        "level2-hidden-layers-slider",
        "level2-save-model-btn",
        "level2-replay-saved-btn",
    ):
        assert control_id in level2_ids
        assert control_id not in level1_ids

    # 4. Assert Level 3 exposes additional controls not present in Level 2.
    level3_ids = collect_component_ids(level3_layout())
    for control_id in (
        "level3-load-data-btn",
        "level3-cell-1-code",
        "level3-cell-6-code",
    ):
        assert control_id in level3_ids
        assert control_id not in level2_ids


# AT-1.1.3.1: Interactive controls and feedback
def test_at_1_1_3_1_interactive_controls_and_feedback():
    # 1. Start on the first interactive learning level.
    level1_component_ids = collect_component_ids(level1_layout())

    # 2. Assert interactive controls are present.
    for control_id in ("l1-dataset", "l1-activation", "l1-epochs", "l1-run-training", "l1-reset"):
        assert control_id in level1_component_ids

    # 3. Assert explanatory boxes are present.
    for box_id in ("l1-architecture-summary", "l1-sample-inspector"):
        assert box_id in level1_component_ids

    # 4. Assert a visual output area is present.
    for output_id in ("l1-decision-boundary", "l1-loss-curve", "l1-architecture-view"):
        assert output_id in level1_component_ids

    # 5. Assert changing a control updates the visual output.
    update_decision_boundary = _original_callback("l1-decision-boundary.figure")
    linear_figure = update_decision_boundary("linear")
    moons_figure = update_decision_boundary("moons")
    assert linear_figure.layout.title.text == "Linear dataset preview"
    assert moons_figure.layout.title.text == "Moons dataset preview"
    assert linear_figure.layout.title.text != moons_figure.layout.title.text


# AT-1.1.3.2: Top bar navigation
def test_at_1_1_3_2_top_bar_navigation():
    # 1. Assert the top bar contains links for Home, Skill Tree, and Sandbox.
    topbar_text = _lower_text(app.layout)
    topbar_hrefs = set(collect_prop_values(app.layout, "href"))
    assert "home" in topbar_text
    assert "skill tree" in topbar_text
    assert "sandbox" in topbar_text
    assert {"/home", "/skilltree", "/sandbox"}.issubset(topbar_hrefs)

    # 2. Assert selecting Home opens the home page.
    assert "home page" in _route_text("/home")

    # 3. Assert selecting Skill Tree opens the skill tree page.
    skill_tree_text = _route_text("/skilltree")
    assert "skill tree" in skill_tree_text
    assert "level 1" in skill_tree_text

    # 4. Assert selecting Sandbox opens the sandbox page.
    sandbox_text = _route_text("/sandbox")
    assert "sandbox" in sandbox_text
    assert "build your own ml pipeline from scratch" in sandbox_text


# AT-1.1.4.1: Visual learning feedback
def test_at_1_1_4_1_visual_learning_feedback():
    # 1. Collect the learner-facing component ids for each level.
    level_component_ids = {
        "level1": collect_component_ids(level1_layout()),
        "level2": collect_component_ids(level2_layout()),
        "level3": collect_component_ids(level3_layout()),
    }

    # 2. Assert each level exposes an FNN architecture visualisation component.
    # 3. Assert each level exposes a training status visualisation component.
    # 4. Assert each level exposes a decision boundary chart component.
    expected_feedback_components = {
        "level1": ("l1-architecture-view", "l1-loss-curve", "l1-decision-boundary"),
        "level2": (
            "level2-network-diagram-graph",
            "level2-training-stage-panel",
            "level2-decision-boundary-graph",
        ),
        "level3": (
            "level3-network-diagram-graph",
            "level3-training-stage-panel",
            "level3-boundary-graph",
        ),
    }

    for level_name, component_ids in level_component_ids.items():
        architecture_id, training_status_id, decision_boundary_id = expected_feedback_components[level_name]
        assert architecture_id in component_ids
        assert training_status_id in component_ids
        assert decision_boundary_id in component_ids