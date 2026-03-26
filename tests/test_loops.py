import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.loops import iterate_until_stable, run_all_chains
from core.types import ChainPath, EventNode


def _node(pos: float, neg: float, prob: float = 1.0, nid: str = "n") -> EventNode:
    return EventNode(id=nid, description=nid, positives=pos, negatives=neg, transition_prob=prob)


def _chain(nodes: List[EventNode], cid: str = "c0") -> ChainPath:
    return ChainPath(chain_id=cid, nodes=nodes)


# ---------------------------------------------------------------------------
# iterate_until_stable — no perturbation
# ---------------------------------------------------------------------------

class TestIterateNoPerturb:
    def test_converges_immediately(self):
        """Without perturbation, score never changes so it should converge after 2 iterations."""
        chain = _chain([_node(10.0, 3.0, 1.0)], "static")
        config = A_S_FLC_Config(max_iterations=5, epsilon=0.01)
        bd, history, stability = iterate_until_stable(chain, config)

        assert history[-1].converged is True
        assert len(history) == 2  # iter 1 sets baseline, iter 2 sees delta=0
        assert stability == pytest.approx(1.0)

    def test_stability_is_one_when_static(self):
        chain = _chain([_node(5.0, 2.0, 0.9), _node(3.0, 1.0, 0.8)], "multi")
        config = A_S_FLC_Config(max_iterations=3)
        _, _, stability = iterate_until_stable(chain, config)
        assert stability == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# iterate_until_stable — with perturbation
# ---------------------------------------------------------------------------

class TestIterateWithPerturb:
    @staticmethod
    def _shift_negatives(chain: ChainPath, iteration: int) -> ChainPath:
        """Increase negatives by 0.5 each iteration to force score drift."""
        new_nodes = [
            EventNode(
                id=n.id,
                description=n.description,
                positives=n.positives,
                negatives=n.negatives + 0.5,
                transition_prob=n.transition_prob,
                children=n.children,
            )
            for n in chain.nodes
        ]
        return ChainPath(chain_id=chain.chain_id, nodes=new_nodes)

    def test_runs_max_iterations(self):
        chain = _chain([_node(10.0, 1.0, 1.0)], "drift")
        config = A_S_FLC_Config(max_iterations=3, epsilon=0.001)
        _, history, _ = iterate_until_stable(chain, config, self._shift_negatives)
        assert len(history) == 3

    def test_net_decreases_with_growing_negatives(self):
        chain = _chain([_node(10.0, 1.0, 1.0)], "drift")
        config = A_S_FLC_Config(max_iterations=3, epsilon=0.001)
        _, history, _ = iterate_until_stable(chain, config, self._shift_negatives)
        nets = [s.net_score for s in history]
        assert nets[0] > nets[-1]

    def test_stability_below_one_when_drifting(self):
        chain = _chain([_node(10.0, 1.0, 1.0)], "drift")
        config = A_S_FLC_Config(max_iterations=3, epsilon=0.001)
        _, _, stability = iterate_until_stable(chain, config, self._shift_negatives)
        assert stability < 1.0

    def test_converges_with_small_perturbation(self):
        """Tiny perturbation within epsilon should still trigger convergence."""
        def tiny_shift(chain: ChainPath, iteration: int) -> ChainPath:
            new_nodes = [
                EventNode(
                    id=n.id, description=n.description,
                    positives=n.positives,
                    negatives=n.negatives + 0.0001,
                    transition_prob=n.transition_prob,
                    children=n.children,
                )
                for n in chain.nodes
            ]
            return ChainPath(chain_id=chain.chain_id, nodes=new_nodes)

        chain = _chain([_node(10.0, 1.0, 1.0)], "tiny")
        config = A_S_FLC_Config(max_iterations=5, epsilon=0.01)
        _, history, _ = iterate_until_stable(chain, config, tiny_shift)
        assert history[-1].converged is True


# ---------------------------------------------------------------------------
# run_all_chains
# ---------------------------------------------------------------------------

class TestRunAllChains:
    def test_sorted_by_net_descending(self):
        chains = [
            _chain([_node(1.0, 0.0, 1.0)], "low"),
            _chain([_node(10.0, 0.0, 1.0)], "high"),
            _chain([_node(5.0, 0.0, 1.0)], "mid"),
        ]
        config = A_S_FLC_Config(buffer_delta=0.0)
        results = run_all_chains(chains, config)
        ids = [r[0].chain_id for r in results]
        assert ids == ["high", "mid", "low"]

    def test_each_result_has_history(self):
        chains = [_chain([_node(5.0, 2.0, 1.0)], "c0")]
        config = A_S_FLC_Config(max_iterations=3)
        results = run_all_chains(chains, config)
        assert len(results) == 1
        _, history, _ = results[0]
        assert len(history) >= 1

    def test_empty_chains_list(self):
        config = A_S_FLC_Config()
        results = run_all_chains([], config)
        assert results == []
