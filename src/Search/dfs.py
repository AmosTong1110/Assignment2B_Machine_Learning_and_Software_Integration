"""Depth-first search for a tree-based route-finding problem (Assignment 2A)."""

from .node import Node


def dfs(problem):
    """Return the first goal node found and the number of Nodes created."""
    origin_x, origin_y = problem.coordinates[problem.origin]
    frontier = [Node(problem.origin, origin_x, origin_y)]
    nodes_created = 1

    while frontier:
        current = frontier.pop()

        if current.node_id in problem.destinations:
            return current, nodes_created

        path_ids = set(current.path())
        children = [
            (child_id, cost)
            for child_id, cost in problem.edges.get(current.node_id, [])
            if child_id not in path_ids
        ]

        children.sort(key=lambda edge: edge[0], reverse=True)
        for child_id, cost in children:
            child_x, child_y = problem.coordinates[child_id]
            frontier.append(
                Node(
                    child_id,
                    child_x,
                    child_y,
                    parent=current,
                    path_cost=current.path_cost + cost,
                )
            )
            nodes_created += 1

    return None, nodes_created
