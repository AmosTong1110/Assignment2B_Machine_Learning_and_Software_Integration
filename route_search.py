"""Search for up to five loopless routes with supplied edge times."""
import heapq, itertools, math
from collections.abc import Callable, Iterable, Mapping
from graph import Edge, Route

def find_routes(
    edges: Iterable[Edge],
    origin: str,
    destination: str,
    hourly_flow: Mapping[str, float],
    edge_minutes: Callable[[Edge, float], float],
    k: int = 5,
    sites: Iterable[str] = (),
) -> list[Route]:
    """Return up to k lowest-cost distinct loopless routes, ordered by minutes.

    ``hourly_flow`` maps site ID to its predicted accumulated hourly volume.
    ``edge_minutes(edge, hourly_flow_at_destination)`` must implement the
    *verified* course conversion and return a nonnegative finite edge time in
    minutes. This function does not estimate or aggregate predictions itself.
    Unknown sites raise ValueError; known but disconnected sites return [].
    For a known site routed to itself, returns one zero-minute route.

    The uniform-cost enumeration is exact for fixed, nonnegative edge times,
    but may be slow on large or highly connected graphs.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 5:
        raise ValueError('k must be an integer from 1 to 5')
    if not callable(edge_minutes):
        raise ValueError('A verified edge_minutes conversion callback is required')
    origin, destination = str(origin).strip(), str(destination).strip()
    adjacency: dict[str, list[Edge]] = {}
    known_sites = {str(s).strip() for s in sites}
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Edge):
            raise ValueError('Each edge must be an Edge')
        if not edge.origin or not edge.destination:
            raise ValueError('Edge endpoints must be nonempty')
        if not math.isfinite(edge.distance_km) or edge.distance_km <= 0:
            raise ValueError('Edge distances must be finite and positive')
        pair = edge.origin, edge.destination
        if pair in seen:
            raise ValueError(f'Duplicate directed edge: {pair!r}')
        seen.add(pair)
        known_sites.update(pair)
        adjacency.setdefault(edge.origin, []).append(edge)
    if origin not in known_sites or destination not in known_sites:
        raise ValueError('Origin or destination is not a known site')
    if origin == destination:
        return [Route((origin,), 0.0)]

    # Cache weights so each edge has a fixed weight for this route search.
    weights: dict[Edge, float] = {}
    counter = itertools.count()
    queue: list[tuple[float, int, tuple[str, ...]]] = [(0.0, next(counter), (origin,))]
    results: list[Route] = []
    while queue and len(results) < k:
        cost, _, path = heapq.heappop(queue)
        if path[-1] == destination:
            results.append(Route(path, cost))
            continue
        for edge in adjacency.get(path[-1], []):
            if edge.destination in path:
                continue
            if edge not in weights:
                if edge.destination not in hourly_flow:
                    raise ValueError(f'Missing hourly flow for site {edge.destination}')
                try:
                    flow = float(hourly_flow[edge.destination])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid hourly flow for site {edge.destination}') from exc
                if not math.isfinite(flow) or flow < 0:
                    raise ValueError(f'Hourly flow must be finite and nonnegative for site {edge.destination}')
                try:
                    minutes = float(edge_minutes(edge, flow))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Invalid travel time for {edge.origin} -> {edge.destination}') from exc
                if not math.isfinite(minutes) or minutes < 0:
                    raise ValueError(f'Travel time must be finite and nonnegative for {edge.origin} -> {edge.destination}')
                weights[edge] = minutes
            heapq.heappush(queue, (cost + weights[edge], next(counter), path + (edge.destination,)))
    return results


