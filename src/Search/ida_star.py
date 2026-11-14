"""Iterative Deepening A* search (CUS2) for the route-finding problem (Assignment 2A).

CUS2 finds a path with the FEWEST MOVES. Here g(n) is the number of
edges from the origin and h(n) is an admissible estimate of the number
of edges still needed, so f(n) = g(n) + h(n) is measured in moves.
"""

import math

from .heuristic import longest_edge_length, nearest_destination_distance
from .node import Node


def ida_star(problem):
    """Return (goal Node or None, number of Nodes created)."""

    # The longest edge (straight-line length) in the map. One move can
    # never get closer to a destination by more than this distance.
    max_edge_length = longest_edge_length(problem)

    def heuristic(node_id):
        """Admissible lower bound on the number of moves still needed."""
        if node_id in problem.destinations:
            return 0
        if max_edge_length == 0:
            return 1  # every move covers no distance, so only "at least 1"
        distance = nearest_destination_distance(problem, node_id)
        # small epsilon avoids floating-point error, e.g. 3.0000000001
        return max(1, math.ceil(distance / max_edge_length - 1e-9))

    origin_x, origin_y = problem.coordinates[problem.origin]
    start = Node(problem.origin, origin_x, origin_y)

    # Origin is counted at the start of every round (same rule as IDDFS).
    nodes_created = 0
    threshold = heuristic(start.node_id)   # measured in MOVES
    path_ids = {start.node_id}

    def bounded_search(current, moves, limit):
        """Return (goal or None, smallest exceeded f-value)."""
        nonlocal nodes_created

        f_value = moves + heuristic(current.node_id)   # moves, not cost
        if f_value > limit:
            return None, f_value

        if current.node_id in problem.destinations:
            return current, math.inf

        next_threshold = math.inf
        neighbours = sorted(
            problem.edges.get(current.node_id, []),
            key=lambda edge: edge[0],
        )

        for child_id, edge_cost in neighbours:
            if child_id in path_ids:
                continue

            child_x, child_y = problem.coordinates[child_id]
            child = Node(
                child_id,
                child_x,
                child_y,
                parent=current,
                path_cost=current.path_cost + edge_cost,  # still reported
            )
            nodes_created += 1

            path_ids.add(child_id)
            goal, exceeded_value = bounded_search(child, moves + 1, limit)
            path_ids.remove(child_id)

            if goal is not None:
                return goal, math.inf

            next_threshold = min(next_threshold, exceeded_value)

        return None, next_threshold

    while True:
        nodes_created += 1
        goal, next_threshold = bounded_search(start, 0, threshold)

        if goal is not None:
            return goal, nodes_created

        if next_threshold == math.inf:
            return None, nodes_created

        threshold = next_threshold
