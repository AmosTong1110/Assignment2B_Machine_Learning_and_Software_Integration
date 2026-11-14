"""Custom search 1: iterative deepening depth-first search (IDDFS) (Assignment 2A)."""

from .node import Node


def depth_limited_search(problem, depth_limit):
    """Run DFS that never goes deeper than depth_limit.

    Returns (goal node or None, nodes created, whether the limit cut
    off any part of the tree).
    """
    origin_x, origin_y = problem.coordinates[problem.origin]
    stack = [(Node(problem.origin, origin_x, origin_y), 0)]
    nodes_created = 1
    cutoff = False

    while stack:
        current, depth = stack.pop()

        if current.node_id in problem.destinations:
            return current, nodes_created, cutoff

        path_ids = set(current.path())
        children = [
            (child_id, cost)
            for child_id, cost in problem.edges.get(current.node_id, [])
            if child_id not in path_ids
        ]

        if depth == depth_limit:
            if children:
                cutoff = True  # there was more to explore below the limit
            continue

        # Reverse order so the smallest node ID is popped first.
        children.sort(key=lambda edge: edge[0], reverse=True)
        for child_id, cost in children:
            child_x, child_y = problem.coordinates[child_id]
            stack.append(
                (
                    Node(
                        child_id,
                        child_x,
                        child_y,
                        parent=current,
                        path_cost=current.path_cost + cost,
                    ),
                    depth + 1,
                )
            )
            nodes_created += 1

    return None, nodes_created, cutoff


def iddfs(problem):
    """Return the first goal node found and the total nodes created.

    Nodes created are counted across all iterations, because each
    iteration rebuilds the tree from the origin.
    """
    nodes_created = 0
    depth_limit = 0

    while True:
        goal, created, cutoff = depth_limited_search(problem, depth_limit)
        nodes_created += created

        if goal is not None:
            return goal, nodes_created
        if not cutoff:
            # The whole tree was explored and no goal exists.
            return None, nodes_created

        depth_limit += 1
