"""Course flow-to-time conversion, using the printed equation in v1.0.

The brief selects DESTINATION flow; the conversion PDF says STARTING flow.
The user chose the brief's destination convention pending tutor confirmation.
The equation below uses its printed coefficient, which differs slightly from
an equation printed on the PDF's graph. Flows above curve capacity are rejected.
"""
import math

A = 1.4648375
B = 93.75
CAPACITY = B * B / (4 * A)


def speed_from_hourly_flow(flow):
    """Return under-capacity speed in km/h, capped at 60 km/h."""
    q=float(flow)
    if not math.isfinite(q) or q < 0 or q > CAPACITY:
        raise ValueError(f'Hourly flow must be finite and between 0 and {CAPACITY:.6f} vehicles/hour.')
    root=(B+math.sqrt(max(0.0,B*B-4*A*q)))/(2*A)
    return min(60.0,root)


def travel_minutes(distance_km, hourly_flow, controlled_intersection):
    """Distance/speed plus 30 seconds only when the intersection is controlled."""
    distance=float(distance_km)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('Distance must be finite and positive.')
    if not isinstance(controlled_intersection,bool):
        raise ValueError('controlled_intersection must be explicitly True or False.')
    return 60*distance/speed_from_hourly_flow(hourly_flow)+(0.5 if controlled_intersection else 0)


def course_edge_minutes(edge, hourly_flow):
    """Require explicit controlled-site metadata; never assume all sites controlled."""
    controlled = getattr(edge, 'controlled_intersection', None)
    if controlled is None:
        raise ValueError('Missing controlled-intersection metadata for destination site.')
    return travel_minutes(edge.distance_km, hourly_flow, controlled)
