"""
travel_time.py  -  traffic flow  ->  speed  ->  travel time   (Task 3, pure maths)
==================================================================================
Implements "Traffic Flow to Travel Time Conversion v1.0.pdf" (assignment_info/).

    travel time of an edge A->B  =  distance(A,B) / speed(flow)  +  30 s intersection delay

Speed comes from a (simplified) fundamental diagram of traffic flow - a parabola:

    flow = A * speed^2 + B * speed        A = -1500 / 32^2 = -1.46484375,  B = 93.75

    * the turning point (capacity) is 1500 veh/h at 32 km/h
    * below capacity there are two speeds per flow; we use the GREEN (uncongested,
      faster) branch, as the PDF assumes every road is under capacity
    * the speed is capped at the 60 km/h speed limit (true for flow <= ~351 veh/h)
    * flow above capacity has no real solution  ->  we clamp it to the capacity speed

NOTE: the PDF text prints A = -1.4648375 (a typo - one digit missing).  The graph in
the same PDF uses -1.46484375 = -1500/32^2, which puts the turning point exactly at
(1500, 32).  With the typo the square root goes negative just below 1500 veh/h.

No ML and no file access here, so every function is easy to unit-test
(tests/test_travel_time.py).  Units: flow in vehicles/HOUR, speed in km/h,
distance in km, time in SECONDS.
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Defaults (config.yaml -> travel_time: overrides these)
# --------------------------------------------------------------------------
SPEED_LIMIT_KMH = 60.0          # same speed limit on every link (assignment assumption i)
CAPACITY_FLOW_VPH = 1500.0      # flow at the turning point of the parabola
CAPACITY_SPEED_KMH = 32.0       # speed at the turning point of the parabola
INTERSECTION_DELAY_S = 30.0     # average delay per controlled intersection (assumption iii)


def parabola_coefficients(capacity_flow: float = CAPACITY_FLOW_VPH,
                          capacity_speed: float = CAPACITY_SPEED_KMH) -> tuple[float, float]:
    """
    Coefficients (A, B) of  flow = A*v^2 + B*v  whose turning point is
    (capacity_speed, capacity_flow).   Default: A = -1.46484375, B = 93.75.
    """
    a = -capacity_flow / capacity_speed ** 2
    b = -2.0 * capacity_speed * a
    return a, b


def speed_to_flow(speed_kmh: float,
                  capacity_flow: float = CAPACITY_FLOW_VPH,
                  capacity_speed: float = CAPACITY_SPEED_KMH) -> float:
    """The PDF formula itself: flow (veh/h) for a given speed (km/h)."""
    a, b = parabola_coefficients(capacity_flow, capacity_speed)
    return a * speed_kmh ** 2 + b * speed_kmh


def flow_to_speed(flow_vph: float,
                  speed_limit: float = SPEED_LIMIT_KMH,
                  capacity_flow: float = CAPACITY_FLOW_VPH,
                  capacity_speed: float = CAPACITY_SPEED_KMH,
                  congested: bool = False) -> float:
    """
    Expected speed (km/h) for a traffic flow (veh/h).

    congested=False -> green branch (under capacity, the default the PDF asks for)
    congested=True  -> red branch   (over capacity: same flow, but slow, jammed traffic)

    Solving  A*v^2 + B*v - flow = 0  for v:
        v = (-B -/+ sqrt(B^2 + 4*A*flow)) / (2A)      ( '-' = green, '+' = red, since A < 0 )
    """
    if flow_vph < 0:
        raise ValueError(f"flow must be >= 0, got {flow_vph}")
    if flow_vph >= capacity_flow:                    # at/over capacity: no real root -> clamp
        return float(capacity_speed)

    a, b = parabola_coefficients(capacity_flow, capacity_speed)
    root = math.sqrt(b * b + 4.0 * a * flow_vph)     # > 0 because flow < capacity
    sign = 1.0 if congested else -1.0
    speed = (-b + sign * root) / (2.0 * a)
    return float(min(speed, speed_limit))            # never faster than the speed limit


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle ('as the crow flies') distance in km between two lat/lon points."""
    r = 6371.0088                                    # mean Earth radius (km)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * r * math.asin(math.sqrt(h))


def edge_travel_time_s(distance_km: float, flow_vph: float,
                       speed_limit: float = SPEED_LIMIT_KMH,
                       capacity_flow: float = CAPACITY_FLOW_VPH,
                       capacity_speed: float = CAPACITY_SPEED_KMH,
                       intersection_delay_s: float = INTERSECTION_DELAY_S,
                       congested: bool = False) -> float:
    """
    Travel time (seconds) along ONE edge A->B:
        driving time  distance / speed(flow) * 3600   +   delay at the next intersection
    """
    if distance_km < 0:
        raise ValueError(f"distance must be >= 0, got {distance_km}")
    speed = flow_to_speed(flow_vph, speed_limit, capacity_flow, capacity_speed, congested)
    return distance_km / speed * 3600.0 + intersection_delay_s


def params_from_config(cfg: dict | None) -> dict:
    """Pick the keyword arguments of edge_travel_time_s() out of config.yaml's travel_time: section."""
    tt = (cfg or {}).get("travel_time", {}) or {}
    return {
        "speed_limit": float(tt.get("speed_limit_kmh", SPEED_LIMIT_KMH)),
        "capacity_flow": float(tt.get("capacity_flow_vph", CAPACITY_FLOW_VPH)),
        "capacity_speed": float(tt.get("capacity_speed_kmh", CAPACITY_SPEED_KMH)),
        "intersection_delay_s": float(tt.get("intersection_delay_s", INTERSECTION_DELAY_S)),
    }
