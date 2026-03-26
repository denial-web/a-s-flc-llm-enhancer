import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.forces import (
    apply_buffer,
    compute_estimated_negatives,
    compute_exact_positives,
    compute_uncertainty_factor,
    rank_chains,
    score_chain,
)
from core.types import ChainPath, EventNode


def _node(pos: float, neg: float, prob: float = 1.0, nid: str = "n") -> EventNode:
    return EventNode(id=nid, description=nid, positives=pos, negatives=neg, transition_prob=prob)


def _chain(nodes: List[EventNode], cid: str = "c0") -> ChainPath:
    return ChainPath(chain_id=cid, nodes=nodes)


# ---------------------------------------------------------------------------
# compute_exact_positives
# ---------------------------------------------------------------------------

class TestComputeExactPositives:
    def test_single_node_prob_1(self):
        nodes = [_node(10.0, 0.0, 1.0)]
        assert compute_exact_positives(nodes) == pytest.approx(10.0)

    def test_single_node_prob_half(self):
        nodes = [_node(10.0, 0.0, 0.5)]
        assert compute_exact_positives(nodes) == pytest.approx(5.0)

    def test_cumulative_probability(self):
        nodes = [_node(10.0, 0.0, 0.5), _node(10.0, 0.0, 0.5)]
        # first: 10 * 0.5 = 5.0
        # second: 10 * 0.5 * 0.5 = 2.5
        assert compute_exact_positives(nodes) == pytest.approx(7.5)

    def test_zero_positives(self):
        nodes = [_node(0.0, 5.0, 1.0), _node(0.0, 3.0, 1.0)]
        assert compute_exact_positives(nodes) == pytest.approx(0.0)

    def test_prob_zero_kills_downstream(self):
        nodes = [_node(10.0, 0.0, 0.0), _node(100.0, 0.0, 1.0)]
        # cumulative prob hits 0 at first node, so both contribute 0
        assert compute_exact_positives(nodes) == pytest.approx(0.0)

    def test_three_node_chain(self):
        nodes = [_node(4.0, 0.0, 0.9), _node(6.0, 0.0, 0.8), _node(2.0, 0.0, 0.7)]
        # cum_prob: 0.9, 0.72, 0.504
        # weighted: 4*0.9=3.6, 6*0.72=4.32, 2*0.504=1.008
        assert compute_exact_positives(nodes) == pytest.approx(8.928)


# ---------------------------------------------------------------------------
# compute_estimated_negatives
# ---------------------------------------------------------------------------

class TestComputeEstimatedNegatives:
    def test_basic_sum(self):
        nodes = [_node(0.0, 3.0), _node(0.0, 2.5)]
        assert compute_estimated_negatives(nodes) == pytest.approx(5.5)

    def test_zero_negatives(self):
        nodes = [_node(5.0, 0.0), _node(3.0, 0.0)]
        assert compute_estimated_negatives(nodes) == pytest.approx(0.0)

    def test_single_node(self):
        nodes = [_node(0.0, 7.7)]
        assert compute_estimated_negatives(nodes) == pytest.approx(7.7)


# ---------------------------------------------------------------------------
# compute_uncertainty_factor
# ---------------------------------------------------------------------------

class TestComputeUncertaintyFactor:
    def test_single_node_returns_1(self):
        nodes = [_node(0.0, 5.0)]
        assert compute_uncertainty_factor(nodes) == pytest.approx(1.0)

    def test_identical_negatives(self):
        nodes = [_node(0.0, 3.0), _node(0.0, 3.0)]
        # variance = 0, so result = 0 + 1.0 = 1.0
        assert compute_uncertainty_factor(nodes) == pytest.approx(1.0)

    def test_varied_negatives(self):
        nodes = [_node(0.0, 1.0), _node(0.0, 5.0)]
        # variance of [1, 5] = 4.0, result = 4.0 + 1.0 = 5.0
        assert compute_uncertainty_factor(nodes) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# apply_buffer
# ---------------------------------------------------------------------------

class TestApplyBuffer:
    def test_default_config(self):
        config = A_S_FLC_Config()  # delta=0.15
        result = apply_buffer(10.0, 1.0, config)
        assert result == pytest.approx(10.0 + 0.15 * 1.0)

    def test_high_uncertainty(self):
        config = A_S_FLC_Config(buffer_delta=0.2)
        result = apply_buffer(5.0, 10.0, config)
        assert result == pytest.approx(5.0 + 0.2 * 10.0)

    def test_zero_delta(self):
        config = A_S_FLC_Config(buffer_delta=0.0)
        result = apply_buffer(5.0, 999.0, config)
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# score_chain
# ---------------------------------------------------------------------------

class TestScoreChain:
    def test_known_chain(self):
        nodes = [_node(8.0, 5.5, 0.95), _node(6.0, 3.0, 0.9)]
        chain = _chain(nodes, "direct")
        config = A_S_FLC_Config(buffer_delta=0.15)

        bd = score_chain(chain, config)
        assert bd.chain_id == "direct"
        assert bd.positives == pytest.approx(
            compute_exact_positives(nodes), abs=1e-3
        )
        expected_est = 5.5 + 3.0
        assert bd.negatives_estimated == pytest.approx(expected_est, abs=1e-3)
        assert bd.net == pytest.approx(bd.positives - bd.negatives_buffered, abs=1e-3)

    def test_single_node_chain(self):
        nodes = [_node(10.0, 2.0, 1.0)]
        chain = _chain(nodes, "solo")
        config = A_S_FLC_Config(buffer_delta=0.15)
        bd = score_chain(chain, config)
        # uncertainty_factor = 1.0 (single node)
        assert bd.negatives_buffered == pytest.approx(2.0 + 0.15 * 1.0, abs=1e-3)
        assert bd.net == pytest.approx(10.0 - bd.negatives_buffered, abs=1e-3)


# ---------------------------------------------------------------------------
# rank_chains
# ---------------------------------------------------------------------------

class TestRankChains:
    def test_descending_order(self):
        config = A_S_FLC_Config(buffer_delta=0.0)
        chains = [
            _chain([_node(1.0, 0.0, 1.0)], "low"),
            _chain([_node(10.0, 0.0, 1.0)], "high"),
            _chain([_node(5.0, 0.0, 1.0)], "mid"),
        ]
        ranked = rank_chains(chains, config)
        assert [b.chain_id for b in ranked] == ["high", "mid", "low"]

    def test_buffer_can_reorder(self):
        """A chain with high positives but high-variance negatives can drop below
        a chain with lower positives but stable negatives after buffering."""
        config = A_S_FLC_Config(buffer_delta=0.5)
        # Chain A: high positives, but negatives vary wildly -> high uncertainty
        chain_a = _chain(
            [_node(10.0, 1.0, 1.0), _node(0.0, 9.0, 1.0)], "volatile"
        )
        # Chain B: moderate positives, uniform negatives -> low uncertainty
        chain_b = _chain(
            [_node(8.0, 3.0, 1.0), _node(0.0, 3.0, 1.0)], "stable"
        )
        ranked = rank_chains([chain_a, chain_b], config)
        assert ranked[0].chain_id == "stable"
