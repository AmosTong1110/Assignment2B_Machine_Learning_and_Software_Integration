"""Heuristics for tree-based search (Assignment 2A, now unit-aware through Problem)."""


def nearest_destination_distance(problem, node_id):
    """Return the straight-line distance to the nearest destination."""
    return min(problem.distance(node_id, destination) for destination in problem.destinations)


def cost_heuristic(problem, node_id):
    """
    Lower bound on the remaining COST to the nearest destination (used by A* and GBFS).

    Part A maps: the cost is a length, so this is just the straight-line distance.
    Boroondara:  straight-line km * 60 s/km = time at the 60 km/h limit with no delays.
                 Every real edge is at least that long and slow, so it never overestimates.
    """
    return nearest_destination_distance(problem, node_id) * problem.cost_per_distance


def longest_edge_length(problem):
    """Return the longest straight-line length of any edge in the graph."""
    longest = 0.0
    for from_id, neighbours in problem.edges.items():
        for to_id, _ in neighbours:
            longest = max(longest, problem.distance(from_id, to_id))
    return longest