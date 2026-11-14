"""A* search for a tree-based route-finding problem (Assignment 2A)."""

import heapq

from .heuristic import cost_heuristic
from .node import Node


def astar(problem):
    """Return the first goal node found and the number of nodes created."""
    origin_x, origin_y = problem.coordinates[problem.origin]
    origin = Node(problem.origin, origin_x, origin_y)

    counter = 0
    frontier = [(cost_heuristic(problem, origin.node_id), origin.node_id, counter, origin)]
    nodes_created = 1

    while frontier:
        _, _, _, current = heapq.heappop(frontier)

        if current.node_id in problem.destinations:
            return current, nodes_created

        path_ids = set(current.path())

        for child_id, cost in problem.edges.get(current.node_id, []):
            if child_id in path_ids:
                continue
            child_x, child_y = problem.coordinates[child_id]
            child = Node(
                child_id,
                child_x,
                child_y,
                parent=current,
                path_cost=current.path_cost + cost,
            )
            counter += 1
            # f = g + h: cost so far plus estimated cost to the goal
            priority = child.path_cost + cost_heuristic(problem, child_id)
            heapq.heappush(frontier, (priority, child_id, counter, child))
            nodes_created += 1

    return None, nodes_created
