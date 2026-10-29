"""Small, explicit routing component for the traffic-guidance prototype.

Inputs are directed links; a two-way street needs one row in each direction.
Travel-time conversion is deliberately supplied by the caller: this module does
not invent the course's traffic-flow-to-travel-time formula. Flow values passed
to the conversion callback represent vehicles per hour at the destination site.
The callback must return total minutes for the edge, including any controlled-
intersection delay applicable under the course conversion method.
"""
from __future__ import annotations

import csv
import heapq
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping


@dataclass(frozen=True)
class Edge:
    origin: str
    destination: str
    distance_km: float


@dataclass(frozen=True)
class Route:
    sites: tuple[str, ...]
    minutes: float


def load_sites(path: str | Path) -> set[str]:
    """Read an optional site list CSV with column site_id, retaining isolates."""
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'site_id' not in reader.fieldnames:
            raise ValueError('Sites CSV must have a site_id column')
        sites = set()
        for row_number, row in enumerate(reader, start=2):
            site = (row.get('site_id') or '').strip()
            if not site:
                raise ValueError(f'Empty site_id on row {row_number}')
            sites.add(site)
        return sites


def load_edges(path: str | Path) -> list[Edge]:
    """Read directed CSV columns origin,destination,distance_km."""
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = {'origin', 'destination', 'distance_km'}
        if not reader.fieldnames or not columns.issubset(reader.fieldnames):
            raise ValueError('Edges CSV requires origin,destination,distance_km')
        edges = []
        seen = set()
        for row_number, row in enumerate(reader, start=2):
            origin = (row.get('origin') or '').strip()
            destination = (row.get('destination') or '').strip()
            try:
                distance = float(row.get('distance_km') or '')
            except ValueError as exc:
                raise ValueError(f'Invalid distance on row {row_number}') from exc
            if not origin or not destination:
                raise ValueError(f'Empty endpoint on row {row_number}')
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError(f'Distance must be finite and positive on row {row_number}')
            if (origin, destination) in seen:
                raise ValueError(f'Duplicate directed edge on row {row_number}')
            seen.add((origin, destination))
            edges.append(Edge(origin, destination, distance))
        return edges


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
