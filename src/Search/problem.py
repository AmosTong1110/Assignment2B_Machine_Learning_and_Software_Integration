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