import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.chains import build_tree_from_llm_output, enumerate_paths
from core.types import EventNode


def _node(nid: str, children: Optional[List[EventNode]] = None, **kw) -> EventNode:
    return EventNode(
        id=nid,
        description=kw.get("description", nid),
        positives=kw.get("positives", 0.0),
        negatives=kw.get("negatives", 0.0),
        transition_prob=kw.get("transition_prob", 1.0),
        children=children or [],
    )


# ---------------------------------------------------------------------------
# enumerate_paths
# ---------------------------------------------------------------------------

class TestEnumeratePaths:
    def test_travel_tree_has_four_paths(self):
        """The canonical travel tree from planning_task.py should yield 4 leaf paths."""
        direct = _node("direct", children=[_node("direct-hotel")], positives=8.0, negatives=5.5, transition_prob=0.95)
        layover_delay = _node("layover-delay", children=[_node("layover-hotel")], negatives=3.5, transition_prob=0.6)
        layover_smooth = _node("layover-smooth", children=[_node("layover-smooth-hotel")], positives=5.0, transition_prob=0.4)
        layover = _node("layover", children=[layover_delay, layover_smooth], positives=4.0, negatives=2.0, transition_prob=0.85)
        train = _node("train", children=[_node("train-hotel")], positives=9.0, negatives=6.0, transition_prob=0.7)
        root = _node("root", children=[direct, layover, train])

        config = A_S_FLC_Config(max_branches=10)
        paths = enumerate_paths(root, config)
        assert len(paths) == 4

    def test_single_leaf_tree(self):
        root = _node("root")
        config = A_S_FLC_Config()
        paths = enumerate_paths(root, config)
        assert len(paths) == 1
        assert paths[0].nodes[0].id == "root"

    def test_linear_chain(self):
        """A -> B -> C (no branching) should produce exactly one path."""
        c = _node("c")
        b = _node("b", children=[c])
        a = _node("a", children=[b])
        config = A_S_FLC_Config()
        paths = enumerate_paths(a, config)
        assert len(paths) == 1
        assert [n.id for n in paths[0].nodes] == ["a", "b", "c"]

    def test_max_branches_limit(self):
        """With max_branches=2, only the first 2 leaf paths should be returned."""
        children = [_node(f"leaf-{i}") for i in range(5)]
        root = _node("root", children=children)
        config = A_S_FLC_Config(max_branches=2)
        paths = enumerate_paths(root, config)
        assert len(paths) == 2

    def test_path_probability_computed(self):
        child = _node("child", transition_prob=0.8)
        root = _node("root", children=[child], transition_prob=0.9)
        config = A_S_FLC_Config()
        paths = enumerate_paths(root, config)
        assert paths[0].path_probability == pytest.approx(0.9 * 0.8)

    def test_path_totals_computed(self):
        child = _node("child", positives=3.0, negatives=1.0)
        root = _node("root", children=[child], positives=5.0, negatives=2.0)
        config = A_S_FLC_Config()
        paths = enumerate_paths(root, config)
        assert paths[0].total_positives == pytest.approx(8.0)
        assert paths[0].total_negatives == pytest.approx(3.0)

    def test_wide_tree(self):
        """Root with 4 direct leaf children -> 4 paths."""
        children = [_node(f"leaf-{i}", positives=float(i)) for i in range(4)]
        root = _node("root", children=children)
        config = A_S_FLC_Config(max_branches=10)
        paths = enumerate_paths(root, config)
        assert len(paths) == 4


# ---------------------------------------------------------------------------
# build_tree_from_llm_output
# ---------------------------------------------------------------------------

class TestBuildTreeFromLlmOutput:
    def test_flat_chains(self):
        raw = [
            {"id": "a", "description": "Option A", "positives": 5.0, "negatives": 2.0},
            {"id": "b", "description": "Option B", "positives": 3.0, "negatives": 1.0},
        ]
        tree = build_tree_from_llm_output(raw)
        assert tree.id == "root"
        assert len(tree.children) == 2
        assert tree.children[0].positives == 5.0

    def test_nested_children(self):
        raw = [
            {
                "id": "a",
                "description": "Option A",
                "children": [
                    {"id": "a1", "description": "Sub-option A1", "positives": 1.0, "negatives": 0.5}
                ],
            }
        ]
        tree = build_tree_from_llm_output(raw)
        assert len(tree.children) == 1
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].id == "a1"

    def test_defaults_applied(self):
        raw = [{"id": "x", "description": "minimal"}]
        tree = build_tree_from_llm_output(raw)
        node = tree.children[0]
        assert node.positives == 0.0
        assert node.negatives == 0.0
        assert node.transition_prob == 1.0
