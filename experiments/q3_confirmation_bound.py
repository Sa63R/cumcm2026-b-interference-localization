"""Independent rational certificates for Q3's guarantee-only movement bound.

No simulator, validation database, source truth, or network access is used.
The optional certificate *generator* uses SciPy; verification uses only the
Python standard library, integer arithmetic, and Fraction.

Scope: an online policy starts at (0, 0), and must certify completion using
legal observations for every allowed configuration. When the actual N < 16,
an extra minimum-reception-radius source on an empty channel is still allowed.
Thus the 1000 m neighbourhood of the *whole trajectory* must cover the 1800 m
arena. This is not a lower bound on the final confirmation tail, and it must
not be imposed when N == 16.

For each of the 720 orders of six fixed arena points c_i, let q_i be a
trajectory's first entry into B(c_i, 1000), in chronological order, q_0=0.
Any vectors u_i with ||u_i|| <= 1, and u_7=0, give

 sum_i ||q_i-q_{i-1}|| >= sum_i (u_i-u_{i+1}) dot c_i
                                 - 1000 sum_i ||u_i-u_{i+1}||.

We verify all vector constraints exactly and round every subtracted norm UP.
The weakest of the 720 verified values is therefore a rigorous lower bound.
Overlapping disks and simultaneous first entries cause no problem: break
ties arbitrarily, and zero-length edges are allowed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path


POINTS = ((1800, 0), (900, 1558), (-900, 1558), (-1800, 0),
          (-900, -1558), (900, -1558))
TARGET_LENGTH = Fraction("4425.6")
CERTIFICATE_PATH = (Path(__file__).resolve().parents[1] / "research" /
                    "q3_fresh_round2" / "confirmation_certificate.json")


def _atan_bounds(inverse: int, terms: int = 40) -> tuple[Fraction, Fraction]:
    """Alternating-series bounds on atan(1/inverse), with exact remainders."""
    x = Fraction(1, inverse)
    partial = sum(((-1) ** k * x ** (2 * k + 1) / (2 * k + 1)
                   for k in range(terms)), Fraction())
    next_term = (-1) ** terms * x ** (2 * terms + 1) / (2 * terms + 1)
    return min(partial, partial + next_term), max(partial, partial + next_term)


def pi_lower_bound() -> Fraction:
    """Machin identity pi = 16 atan(1/5) - 4 atan(1/239)."""
    low5, _ = _atan_bounds(5)
    _, high239 = _atan_bounds(239)
    return 16 * low5 - 4 * high239


def area_length_lower_bound() -> Fraction:
    """The tubular-area argument: L >= pi * (1800**2-1000**2)/2000."""
    return pi_lower_bound() * 1120


def sqrt_upper(value: Fraction, scale: int = 10 ** 12) -> Fraction:
    """An exact upper bound on sqrt(value), at most 1/scale too large."""
    value = Fraction(value)
    if value < 0 or not isinstance(scale, int) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("Nonnegative radicand and positive integer scale required")
    numerator = value.numerator * scale * scale
    denominator = value.denominator
    root = math.isqrt(numerator // denominator)
    if root * root * denominator < numerator:
        root += 1
    return Fraction(root, scale)


def _require_int(value, description):
    if type(value) is not int:
        raise ValueError(f"{description} must be an integer")
    return value


def verify_order(record: dict, scale: int) -> Fraction:
    """Verify one rational dual certificate; return its conservative value."""
    _require_int(scale, "Dual scale")
    if scale <= 0:
        raise ValueError("Dual scale must be positive")
    order = record.get("order")
    if (not isinstance(order, list) or len(order) != 6 or
            any(type(x) is not int for x in order) or sorted(order) != list(range(6))):
        raise ValueError("Order must be a permutation of all six point indices")
    numerators = record.get("edge_dual_numerators")
    if not isinstance(numerators, list) or len(numerators) != 6:
        raise ValueError("Exactly six edge dual vectors are required")
    vectors = []
    for vector in numerators:
        if not isinstance(vector, list) or len(vector) != 2:
            raise ValueError("Every edge dual must have two integer components")
        x, y = [_require_int(v, "Dual component") for v in vector]
        if x * x + y * y > scale * scale:
            raise ValueError("Infeasible dual vector has norm greater than one")
        vectors.append((Fraction(x, scale), Fraction(y, scale)))
    vectors.append((Fraction(), Fraction()))  # Free endpoint: u_7=0.
    value = Fraction()
    for i, index in enumerate(order):
        dx, dy = (vectors[i][axis] - vectors[i + 1][axis] for axis in (0, 1))
        cx, cy = POINTS[index]
        value += dx * cx + dy * cy - 1000 * sqrt_upper(dx * dx + dy * dy)
    return value


def verify_certificate(data: dict) -> dict:
    """Check geometry, exact dual feasibility, target, and complete 720 orders.

    Raises ValueError for missing/corrupt/insufficient evidence. Solver status,
    precomputed objective values, and a claimed optimum are never trusted.
    """
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported certificate schema")
    if (data.get("domain_radius_m") != 1800 or data.get("coverage_radius_m") != 1000 or
            data.get("origin") != [0, 0] or data.get("points") != [list(p) for p in POINTS]):
        raise ValueError("Certificate geometry differs from the proven arena")
    if any(x * x + y * y > 1800 ** 2 for x, y in POINTS):
        raise ValueError("A required point lies outside the arena")
    target = Fraction(data.get("target_length_m", "0"))
    if target != TARGET_LENGTH:
        raise ValueError("Certificate does not prove the required 4425.6 m target")
    records = data.get("certificates")
    if not isinstance(records, list) or len(records) != math.factorial(6):
        raise ValueError("All 720 visit orders must be explicitly certified")
    seen = set()
    minimum = None
    weakest_order = None
    for record in records:
        value = verify_order(record, data.get("dual_scale"))
        order = tuple(record["order"])
        if order in seen:
            raise ValueError("Duplicate visit order")
        seen.add(order)
        if value < target:
            raise ValueError(f"Order {order} fails the target: {float(value):.12f} m")
        if minimum is None or value < minimum:
            minimum, weakest_order = value, order
    if seen != set(itertools.permutations(range(6))):
        raise ValueError("Certificate set does not cover all visit orders")
    return {
        "verified_orders": len(seen),
        "certified_length_m": str(target),
        "minimum_verified_dual_m": str(minimum),
        "minimum_verified_dual_decimal_m": float(minimum),
        "weakest_order": list(weakest_order),
        "verification_arithmetic": "Fraction; integer isqrt; all norm penalties rounded upward",
    }


@lru_cache(maxsize=4)
def _verified_file_bound(path: str, modified_ns: int, size: int) -> Fraction:
    # Metadata is part of the cache key: a subsequently replaced proof must
    # be verified again rather than inheriting the previous file's status.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    verify_certificate(data)
    return Fraction(data["target_length_m"])


def confirmation_length_lower_bound(n: int, *, certificate_path=None) -> float:
    """Guarantee-only bound for actual source count n; zero for n == 16.

    Without a certificate file, return the rigorously derived area bound.
    A present but invalid certificate raises: corrupted proof is never accepted.
    The result is rounded toward minus infinity on conversion to binary float.
    """
    if type(n) is not int or not 10 <= n <= 16:
        raise ValueError("Q3 source count must be an integer in [10, 16]")
    if n == 16:
        return 0.0
    path = CERTIFICATE_PATH if certificate_path is None else Path(certificate_path)
    bound = area_length_lower_bound()
    if path.is_file():
        stat = path.stat()
        bound = max(bound, _verified_file_bound(str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return math.nextafter(float(bound), -math.inf)


def empty_channel_action_lower_bound() -> int:
    """Exact conservative proof of the 30 s per empty channel action bound.

    A disk of radius 1000 cuts at most 2 asin(5/9) < 6/5 radians from the
    arena circumference; a failed-clear disk cuts at most 2 asin(1/90) < 1/40.
    The inequalities follow from sin(x) >= x-x^3/6 at x=3/5 and 1/80.
    Every integer mix costing < 30 seconds covers strictly less than 2*pi.
    Switching costs are omitted, which can only weaken the lower bound.
    """
    for angle, ratio in ((Fraction(3, 5), Fraction(5, 9)),
                         (Fraction(1, 80), Fraction(1, 90))):
        if angle - angle ** 3 / 6 <= ratio:
            raise AssertionError("Invalid conservative arc estimate")
    circumference_angle = 2 * pi_lower_bound()
    for measures in range(6):
        for failures in range(10):
            if 5 * measures + 3 * failures < 30:
                arc_bound = measures * Fraction(6, 5) + failures * Fraction(1, 40)
                if arc_bound >= circumference_angle:
                    raise AssertionError("Insufficient empty-channel action proof")
    return 30


def guarantee_time_lower_bound(n: int, source_visit_length_m: float, *, certificate_path=None) -> float:
    """Combine overlapping movement requirements by MAX, never by addition."""
    if not math.isfinite(source_visit_length_m) or source_visit_length_m < 0:
        raise ValueError("Source-visit lower bound must be finite and nonnegative")
    confirmation = confirmation_length_lower_bound(n, certificate_path=certificate_path)
    actions = 5 * n + (empty_channel_action_lower_bound() * (20 - n) if n < 16 else 0)
    exact = max(Fraction(source_visit_length_m), Fraction(confirmation)) / 5 + actions
    return math.nextafter(float(exact), -math.inf)


def _quantize_vectors(vectors, scale):
    """Strictly feasible rational vectors, regardless of numerical solve status."""
    result = []
    for x, y in vectors:
        norm = math.hypot(float(x), float(y))
        shrink = (1 - 1e-10) / max(1.0, norm)
        a, b = round(float(x) * shrink * scale), round(float(y) * shrink * scale)
        while a * a + b * b > scale * scale:
            if abs(a) >= abs(b):
                a -= 1 if a > 0 else -1
            else:
                b -= 1 if b > 0 else -1
        result.append([a, b])
    return result


def _solve_order(order, scale):
    """Small smooth concave dual optimization; solver is only a witness finder."""
    import numpy as np
    from scipy.optimize import minimize

    centres = np.array([POINTS[i] for i in order], dtype=float) / 1000
    edges = np.diff(np.vstack((np.zeros((1, 2)), centres)), axis=0)
    current = edges / np.linalg.norm(edges, axis=1)[:, None]
    # Difference operator (u_i-u_{i+1}), with last u_{i+1}=0.
    def differences(u):
        return np.vstack((u[:-1] - u[1:], u[-1:]))

    def value_gradient(flat, epsilon):
        u = flat.reshape(6, 2)
        d = differences(u)
        norms = np.sqrt(np.sum(d * d, axis=1) + epsilon ** 2)
        objective = float(np.sum(norms) - np.sum(d * centres))
        dd = d / norms[:, None] - centres
        gradient = dd.copy()
        gradient[1:] -= dd[:-1]
        return objective, gradient.ravel()

    def constraints(flat):
        return 1 - np.sum(flat.reshape(6, 2) ** 2, axis=1)

    def constraints_jac(flat):
        jac = np.zeros((6, 12))
        for i in range(6):
            jac[i, 2 * i:2 * i + 2] = -2 * flat[2 * i:2 * i + 2]
        return jac

    best = None
    best_value = -math.inf
    iterations = 0
    for epsilon in (1e-3, 1e-5, 1e-7, 1e-9):
        result = minimize(lambda x: value_gradient(x, epsilon), current.ravel(),
                          method="SLSQP", jac=True,
                          constraints={"type": "ineq", "fun": constraints, "jac": constraints_jac},
                          options={"ftol": 1e-12, "maxiter": 1600})
        iterations += int(result.nit)
        current = result.x.reshape(6, 2)
        record = {"order": list(order), "edge_dual_numerators": _quantize_vectors(current, scale)}
        value = verify_order(record, scale)
        if value > best_value:
            best, best_value = record, value
        if value >= TARGET_LENGTH:
            return best, iterations
    raise RuntimeError(f"No adequate witness for {order}; best verified {float(best_value):.12f} m")


def generate_certificate(output: Path) -> dict:
    """Generate on the authorized remote CPU host; then verify independently.

    Integer centres have only axis-reflection symmetry, not exact 60-degree
    rotation symmetry. Solve 180 representatives, expand to all 720 orders.
    """
    scale = 10 ** 12
    transforms = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
    mappings = [[POINTS.index((sx * x, sy * y)) for x, y in POINTS]
                for sx, sy in transforms]
    records = {}
    solved = 0
    total_iterations = 0
    for order in itertools.permutations(range(6)):
        if order in records:
            continue
        record, iterations = _solve_order(order, scale)
        solved += 1
        total_iterations += iterations
        for (sx, sy), mapping in zip(transforms, mappings):
            transformed_order = tuple(mapping[i] for i in order)
            transformed = {
                "order": list(transformed_order),
                "edge_dual_numerators": [[sx * x, sy * y] for x, y in record["edge_dual_numerators"]],
            }
            if verify_order(transformed, scale) < TARGET_LENGTH:
                raise AssertionError("A symmetry-transformed witness failed verification")
            records[transformed_order] = transformed
        if solved % 10 == 0 or len(records) == 720:
            print(json.dumps({"solved_representatives": solved, "certified_orders": len(records),
                              "solver_iterations": total_iterations}), flush=True)
    data = {
        "schema_version": 1,
        "domain_radius_m": 1800,
        "coverage_radius_m": 1000,
        "origin": [0, 0],
        "points": [list(p) for p in POINTS],
        "target_length_m": str(TARGET_LENGTH),
        "dual_scale": scale,
        "scope": "Complete guaranteed online trajectory, actual source count 10..15; not confirmation tail",
        "generation": {"method": "SLSQP smoothed dual; exact rational acceptance",
                       "solved_representatives": solved, "solver_iterations": total_iterations,
                       "symmetries": "x/y reflections only; all 720 expanded orders verified"},
        "certificates": [records[order] for order in sorted(records)],
    }
    verification = verify_certificate(data)
    data["verification_summary"] = verification
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="Requires SciPy; CPU witness generation")
    parser.add_argument("--output", type=Path, default=CERTIFICATE_PATH)
    args = parser.parse_args()
    if args.generate:
        report = generate_certificate(args.output)
    else:
        report = verify_certificate(json.loads(args.output.read_text(encoding="utf-8")))
    report["area_length_lower_bound_m"] = float(area_length_lower_bound())
    report["empty_channel_action_lower_bound_s"] = empty_channel_action_lower_bound()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
