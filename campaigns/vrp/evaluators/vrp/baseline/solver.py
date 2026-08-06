"""Deadline-aware construction and local-search seed for CVRPTW."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence


# EVOLVE-BLOCK-START
def solve(
    instance: Mapping[str, object],
    deadline: float | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Build routes with savings and nearest-neighbor starts, then improve them."""

    raw_depot = instance["depot"]
    raw_customers = instance["customers"]
    if not isinstance(raw_depot, Mapping) or not isinstance(raw_customers, Sequence):
        raise TypeError("instance depot and customers have invalid shapes")

    def read_stop(
        raw: Mapping[str, object],
    ) -> tuple[int, float, float, float, float, float, float]:
        return (
            int(raw["id"]),
            float(raw["x"]),
            float(raw["y"]),
            float(raw["demand"]),
            float(raw["earliest"]),
            float(raw["latest"]),
            float(raw["service"]),
        )

    depot = read_stop(raw_depot)
    customers = tuple(read_stop(raw) for raw in raw_customers if isinstance(raw, Mapping))
    if len(customers) != len(raw_customers):
        raise TypeError("every customer must be a mapping")
    stops = {depot[0]: depot, **{customer[0]: customer for customer in customers}}
    customer_ids = tuple(customer[0] for customer in customers)
    capacity = float(instance["capacity"])
    vehicle_limit = int(instance["vehicle_limit"])
    rng = random.Random(seed)

    def distance(left_id: int, right_id: int) -> float:
        left = stops[left_id]
        right = stops[right_id]
        return math.hypot(left[1] - right[1], left[2] - right[2])

    def route_distance(route: Sequence[int]) -> float:
        previous = 0
        total = 0.0
        for customer_id in route:
            total += distance(previous, customer_id)
            previous = customer_id
        return total + distance(previous, 0)

    def reported_distance(routes: Sequence[Sequence[int]]) -> float:
        total = sum(route_distance(route) for route in routes)
        return math.floor(total * 100.0 + 0.5) / 100.0

    def feasible(route: Sequence[int]) -> bool:
        load = sum(stops[customer_id][3] for customer_id in route)
        if load > capacity + 1e-9:
            return False
        clock = depot[4] + depot[6]
        previous = 0
        for customer_id in (*route, 0):
            stop = stops[customer_id]
            clock = max(clock + distance(previous, customer_id), stop[4])
            if clock > stop[5] + 1e-9:
                return False
            clock += stop[6]
            previous = customer_id
        return True

    def objective(routes: Sequence[Sequence[int]]) -> tuple[int, float]:
        return (len(routes), reported_distance(routes))

    def savings_start() -> list[list[int]] | None:
        routes = {customer_id: [customer_id] for customer_id in customer_ids}
        owner = {customer_id: customer_id for customer_id in customer_ids}
        savings = [
            (
                distance(0, left) + distance(0, right) - distance(left, right),
                rng.random(),
                left,
                right,
            )
            for left in customer_ids
            for right in customer_ids
            if left < right
        ]
        savings.sort(reverse=True)
        for _, _, left, right in savings:
            left_owner = owner[left]
            right_owner = owner[right]
            if left_owner == right_owner:
                continue
            left_route = routes[left_owner]
            right_route = routes[right_owner]
            candidates: list[list[int]] = []
            for first in (left_route, list(reversed(left_route))):
                if first[-1] != left:
                    continue
                for second in (right_route, list(reversed(right_route))):
                    if second[0] == right:
                        candidates.append([*first, *second])
            for first in (right_route, list(reversed(right_route))):
                if first[-1] != right:
                    continue
                for second in (left_route, list(reversed(left_route))):
                    if second[0] == left:
                        candidates.append([*first, *second])
            merged = next((candidate for candidate in candidates if feasible(candidate)), None)
            if merged is None:
                continue
            routes[left_owner] = merged
            del routes[right_owner]
            for customer_id in merged:
                owner[customer_id] = left_owner
        result = list(routes.values())
        if len(result) <= vehicle_limit and all(feasible(route) for route in result):
            return result
        return None

    def nearest_start(randomized: bool) -> list[list[int]] | None:
        unserved = set(customer_ids)
        routes: list[list[int]] = []
        while unserved:
            route: list[int] = []
            while True:
                previous = route[-1] if route else 0
                choices = [
                    customer_id
                    for customer_id in unserved
                    if feasible([*route, customer_id])
                ]
                if not choices:
                    break
                choices.sort(
                    key=lambda customer_id: (
                        stops[customer_id][5],
                        distance(previous, customer_id),
                        customer_id,
                    )
                )
                choice_pool = choices[: min(4, len(choices))] if randomized else choices[:1]
                chosen = rng.choice(choice_pool)
                route.append(chosen)
                unserved.remove(chosen)
            if not route:
                return None
            routes.append(route)
            if len(routes) > vehicle_limit:
                return None
        return routes

    starts = [route_set for route_set in (savings_start(), nearest_start(False)) if route_set]
    for _ in range(12):
        if deadline is not None and time.monotonic() >= deadline:
            break
        candidate = nearest_start(True)
        if candidate is not None:
            starts.append(candidate)

    if not starts:
        singletons = [[customer_id] for customer_id in customer_ids]
        if len(singletons) > vehicle_limit or not all(feasible(route) for route in singletons):
            raise ValueError("baseline could not construct a feasible incumbent")
        starts.append(singletons)

    incumbent = min(starts, key=objective)
    incumbent_key = objective(incumbent)
    local_deadline = time.monotonic() + min(4.0, max(0.25, len(customer_ids) / 100.0))
    if deadline is not None:
        local_deadline = min(local_deadline, deadline)

    iterations = 0
    while iterations < 5_000 and time.monotonic() < local_deadline:
        iterations += 1
        proposal = [list(route) for route in incumbent]
        move = rng.randrange(3)
        if move == 0 and len(proposal) >= 2:
            left_index, right_index = rng.sample(range(len(proposal)), 2)
            variants = (
                [*proposal[left_index], *proposal[right_index]],
                [*proposal[left_index], *reversed(proposal[right_index])],
                [*reversed(proposal[left_index]), *proposal[right_index]],
            )
            merged = next((route for route in variants if feasible(route)), None)
            if merged is None:
                continue
            proposal[left_index] = merged
            del proposal[right_index]
        elif move == 1 and len(proposal) >= 2:
            source_index, target_index = rng.sample(range(len(proposal)), 2)
            source = proposal[source_index]
            target = proposal[target_index]
            customer_position = rng.randrange(len(source))
            customer_id = source.pop(customer_position)
            insertion = rng.randrange(len(target) + 1)
            target.insert(insertion, customer_id)
            if (source and not feasible(source)) or not feasible(target):
                continue
            if not source:
                del proposal[source_index]
        else:
            route_index = rng.randrange(len(proposal))
            route = proposal[route_index]
            if len(route) < 3:
                continue
            left, right = sorted(rng.sample(range(len(route)), 2))
            if right - left < 2:
                continue
            changed = [*route[:left], *reversed(route[left:right]), *route[right:]]
            if not feasible(changed):
                continue
            proposal[route_index] = changed

        proposal_key = objective(proposal)
        if proposal_key < incumbent_key:
            incumbent = proposal
            incumbent_key = proposal_key

    return {"routes": [[0, *route, 0] for route in incumbent]}
# EVOLVE-BLOCK-END
