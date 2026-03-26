# A-S-FLC Event-Chain Tree Builder
#
# Event chains = path-dependent branches (one action → locked sequence of follow-on events).
# Uses BFS to enumerate all root-to-leaf paths up to max_branches.

from typing import List

from config import A_S_FLC_Config
from core.types import ChainPath, EventNode


def enumerate_paths(
    root: EventNode,
    config: A_S_FLC_Config,
) -> List[ChainPath]:
    """BFS traversal of the event tree, returning up to max_branches leaf-to-root paths."""
    paths: List[ChainPath] = []
    queue: List[List[EventNode]] = [[root]]

    while queue and len(paths) < config.max_branches:
        current_path = queue.pop(0)
        tip = current_path[-1]

        if not tip.children:
            cum_prob = 1.0
            for node in current_path:
                cum_prob *= node.transition_prob
            paths.append(
                ChainPath(
                    chain_id=f"chain-{len(paths)}",
                    nodes=current_path,
                    total_positives=sum(n.positives for n in current_path),
                    total_negatives=sum(n.negatives for n in current_path),
                    path_probability=cum_prob,
                )
            )
        else:
            for child in tip.children:
                queue.append(current_path + [child])

    return paths


def build_tree_from_llm_output(raw_chains: List[dict]) -> EventNode:
    """Convert LLM-generated chain descriptions into an EventNode tree.

    Expects a list of dicts with keys:
        id, description, positives, negatives, transition_prob, children (optional)
    """
    def _parse(data: dict) -> EventNode:
        children = [_parse(c) for c in data.get("children", [])]
        return EventNode(
            id=data["id"],
            description=data["description"],
            positives=data.get("positives", 0.0),
            negatives=data.get("negatives", 0.0),
            transition_prob=data.get("transition_prob", 1.0),
            children=children,
        )

    virtual_root = EventNode(
        id="root",
        description="Decision root",
        children=[_parse(c) for c in raw_chains],
    )
    return virtual_root
