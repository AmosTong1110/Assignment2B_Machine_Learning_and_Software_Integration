"""
Input data for the tree-based search algorithms.

Part A stored a map as  coordinates {id: (x, y)}  +  edges {id: [(neighbour, cost), ...]}.
That is kept.  Two optional settings make the same Problem work for the Boroondara map:

    distance          straight-line distance between two nodes
                      (Part A: Euclidean on (x, y); TBRGS: haversine km on (lat, lon))
    cost_per_distance converts that distance into a LOWER BOUND on the edge cost
                      (Part A: 1, cost is a length; TBRGS: 60 s per km = driving at the
                      60 km/h speed limit with no intersection delay, so A* stays optimal)
"""
import math


def euclidean(a, b):
    """Straight-line distance between two (x, y) points (Part A maps)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Problem:
    """The graph and start/goal data for one route-finding problem."""

    def __init__(self, coordinates, edges, origin, destinations,
                 distance=euclidean, cost_per_distance=1.0):
        self.coordinates = coordinates
        self.edges = edges
        self.origin = origin
        self.destinations = destinations
        self.distance_fn = distance
        self.cost_per_distance = cost_per_distance

    def distance(self, a, b):
        """Straight-line distance between nodes a and b."""
        return self.distance_fn(self.coordinates[a], self.coordinates[b])

    def with_changes(self, origin=None, edges=None):
        """A copy with a new origin and/or edge list (used by Yen's top-k algorithm)."""
        return Problem(self.coordinates, self.edges if edges is None else edges,
                       self.origin if origin is None else origin, self.destinations,
                       self.distance_fn, self.cost_per_distance)


def parse_number(text):
    """Parse whole numbers as ints and other numbers as floats."""
    number = float(text)
    return int(number) if number.is_integer() else number


def load_problem(filename):
    """Read a Part A problem file and return its parsed data (unchanged from Assignment 2A)."""
    sections = {"Nodes": [], "Edges": [], "Origin": [], "Destinations": []}
    current_section = None

    with open(filename, "r", encoding="utf-8-sig") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line:
                continue

            if line in ("Nodes:", "Edges:", "Origin:", "Destinations:"):
                current_section = line[:-1]
            elif current_section is not None:
                sections[current_section].append(line)

    # Store each node ID with its (x, y) coordinates.
    coordinates = {}
    for line in sections["Nodes"]:
        node_text, coordinate_text = line.split(":", 1)
        x_text, y_text = coordinate_text.strip().strip("()").split(",")
        coordinates[int(node_text.strip())] = (parse_number(x_text), parse_number(y_text))

    # Store outgoing edges as (neighbour ID, cost), sorted by neighbour ID.
    edges = {}
    for line in sections["Edges"]:
        edge_text, cost_text = line.split(":", 1)
        from_text, to_text = edge_text.strip().strip("()").split(",")
        from_id = int(from_text)
        to_id = int(to_text)
        if from_id not in coordinates or to_id not in coordinates:
            missing_id = from_id if from_id not in coordinates else to_id
            raise ValueError(f"Edge refers to unknown node ID: {missing_id}")
        cost = parse_number(cost_text.strip())
        edges.setdefault(from_id, []).append((to_id, cost))

    for neighbours in edges.values():
        neighbours.sort(key=lambda edge: edge[0])

    if not sections["Origin"]:
        raise ValueError("Origin section is missing or empty")
    if not sections["Destinations"]:
        raise ValueError("Destinations section is missing or empty")

    origin = int(sections["Origin"][0])
    if origin not in coordinates:
        raise ValueError(f"Origin node ID is not in Nodes section: {origin}")

    destinations = sorted(
        int(destination.strip())
        for destination_line in sections["Destinations"]
        for destination in destination_line.split(";")
        if destination.strip()
    )
    if not destinations:
        raise ValueError("Destinations section is missing or empty")
    for destination in destinations:
        if destination not in coordinates:
            raise ValueError(
                f"Destination node ID is not in Nodes section: {destination}"
            )

    return Problem(coordinates, edges, origin, destinations)
