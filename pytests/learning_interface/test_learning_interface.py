from adaptiveLearning.gamification.skillTree import skilltree_layout
from pages.homePage.layout import home_layout
from pages.levels.level2.layout import level2_layout
from pages.levels.level3.layout import level3_layout
from pytests.helpers import collect_component_ids, collect_prop_values, collect_text


def test_home_layout_contains_dashboard_welcome_copy():
    layout = home_layout()

    text = collect_text(layout)

    assert "Welcome to Neural Network Dashboard" in text
    assert "Skill tree" in text


def test_skill_tree_layout_exposes_level_navigation_links():
    layout = skilltree_layout()

    hrefs = collect_prop_values(layout, "href")

    assert "/level1" in hrefs
    assert "/level2" in hrefs
    assert "/level3" in hrefs


def test_level2_layout_exposes_key_learning_controls():
    layout = level2_layout()

    component_ids = collect_component_ids(layout)

    assert "level2-dataset-dropdown" in component_ids
    assert "level2-hidden-layers-slider" in component_ids
    assert "level2-train-toggle-btn" in component_ids
    assert "level2-reset-btn" in component_ids


def test_level3_layout_exposes_code_cells_and_live_architecture_outputs():
    layout = level3_layout()

    component_ids = collect_component_ids(layout)

    assert "level3-cell-1-code" in component_ids
    assert "level3-cell-6-code" in component_ids
    assert "level3-network-diagram-graph" in component_ids
    assert "level3-boundary-graph" in component_ids