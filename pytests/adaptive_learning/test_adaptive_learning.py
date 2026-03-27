from adaptiveLearning.gamification.skillTree import SKILL_TREE_DATA
from pages.levels.level3.layout import LEVEL3_CELL_SNIPPETS


def test_skill_tree_prerequisites_reference_known_nodes():
    known_nodes = {node["id"] for node in SKILL_TREE_DATA["nodes"]}

    for node_id, prerequisites in SKILL_TREE_DATA["prereqs"].items():
        assert node_id in known_nodes
        assert set(prerequisites).issubset(known_nodes)


def test_skill_tree_contains_a_single_unlocked_starting_node():
    unlocked_nodes = [node for node in SKILL_TREE_DATA["nodes"] if node["unlocked"]]

    assert len(unlocked_nodes) == 1
    assert unlocked_nodes[0]["id"] == "level1"


def test_level3_progression_scaffold_contains_six_ordered_cells():
    assert list(LEVEL3_CELL_SNIPPETS) == [1, 2, 3, 4, 5, 6]


def test_level3_progression_scaffold_starts_with_dataset_and_ends_with_training_config():
    assert "Dataset handling" in LEVEL3_CELL_SNIPPETS[1]
    assert "Training configuration" in LEVEL3_CELL_SNIPPETS[6]