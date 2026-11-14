"""
top_k.py  -  up to k different routes from O to D with any Part A search method   (Task 4)
==========================================================================================
Part A searches stop at the FIRST goal they reach, so they return one route.
Yen's algorithm (Yen, 1971) turns any one-route search into a k-routes search:

    1. route 1 = search(problem)
    2. for every node i on the last accepted route ("spur node"):
         - keep the route up to i ("root path")
         - block the edge leaving i that an accepted route with the same root already used
         - block the root-path nodes before i (so the new route has no loops)
         - search again from i  ->  candidate = root path + new spur path
    3. accept the cheapest candidate; repeat until k routes or no candidates are left

With A* (optimal, admissible heuristic) the result is the true k shortest loop-free routes,
cheapest first.  With BFS/DFS/GBFS/IDDFS/IDA* the routes are valid and all different but
not necessarily the k cheapest; they are sorted by cost before they are returned.
"""
import heapq


def route_cost(edges, path):
    """Total cost of a path, from the problem's edge lists."""
    lookup = {(a, b): c for a, nbrs in edges.items() for b, c in nbrs}
    return sum(lookup[(a, b)] for a, b in zip(path, path[1:]))


def k_routes(problem, search, k=5):
    """
    Returns (routes, nodes_created):
        routes         list of (path, cost), at most k, cheapest first
        nodes_created  total search-tree nodes created over all calls (for comparing methods)
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    goal, nodes_created = search(problem)
    if goal is None:
        return [], nodes_created

    first = goal.path()
    accepted = [(first, route_cost(problem.edges, first))]
    seen = {tuple(first)}
    candidates = []                     # heap of (cost, tie-break counter, path)
    counter = 0

    while len(accepted) < k:
        last = accepted[-1][0]
        for i in range(len(last) - 1):
            spur, root = last[i], last[: i + 1]

            # edges already used by accepted routes that share this root path
            blocked_edges = {(p[i], p[i + 1]) for p, _ in accepted
                             if len(p) > i + 1 and p[: i + 1] == root}
            blocked_nodes = set(root[:-1])
            edges = {
                a: [(b, c) for b, c in nbrs if (a, b) not in blocked_edges and b not in blocked_nodes]
                for a, nbrs in problem.edges.items() if a not in blocked_nodes
            }

            spur_goal, created = search(problem.with_changes(origin=spur, edges=edges))
            nodes_created += created
            if spur_goal is None:
                continue
            path = root[:-1] + spur_goal.path()
            if tuple(path) in seen:
                continue
            seen.add(tuple(path))
            counter += 1
            heapq.heappush(candidates, (route_cost(problem.edges, path), counter, path))

        if not candidates:
            break                       # fewer than k different routes exist
        cost, _, path = heapq.heappop(candidates)
        accepted.append((path, cost))

    accepted.sort(key=lambda route: route[1])
    return accepted, nodes_created
