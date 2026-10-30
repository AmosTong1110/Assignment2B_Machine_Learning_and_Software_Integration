"""Read road nodes and directed links."""
import csv, math
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Edge:
    origin: str
    destination: str
    distance_km: float
    controlled_intersection: bool | None = None


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
            controlled_raw = (row.get('controlled_intersection') or '').strip().lower()
            if controlled_raw not in ('', 'true', 'false', '1', '0'):
                raise ValueError(f'Invalid controlled_intersection on row {row_number}')
            controlled = None if not controlled_raw else controlled_raw in ('true', '1')
            edges.append(Edge(origin, destination, distance, controlled))
        return edges


