"""Exact numeric/exception equivalence to the immutable 760 cover oracle.

These are synthetic geometry fixtures, not simulator cases. The reference is
loaded from its Git blob, never from another mutable working directory.
"""

import hashlib
import math
from pathlib import Path
import random
import subprocess
import types

import pytest

from planning.disk_cover import certifies_disk_cover, disk_cover_radius
from simulator_client.state import Position


BASE_COMMIT = "760af8230a8e5bda6050663b19d871a9701c9829"
BASE_SOURCE_SHA256 = "8731a6b7b8d396de04e2d6155fa130626b218e085672e6c432eff11311d4e09d"


@pytest.fixture(scope="module")
def frozen_cover():
    source = subprocess.run(
        ["git", "show", f"{BASE_COMMIT}:src/planning/disk_cover.py"],
        cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(source).hexdigest() == BASE_SOURCE_SHA256
    module = types.ModuleType("frozen_disk_cover_760")
    exec(compile(source, f"{BASE_COMMIT}:src/planning/disk_cover.py", "exec"), module.__dict__)
    return module


def assert_identical(actual, expected):
    # hex also distinguishes signed zero and catches single-ulp differences.
    assert type(actual) is type(expected)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual.hex() == expected.hex()


def compare(frozen_cover, stations, arena_radius=1800.0):
    expected = frozen_cover.disk_cover_radius(stations, arena_radius)
    actual = disk_cover_radius(stations, arena_radius)
    assert_identical(actual, expected)
    return actual


@pytest.mark.parametrize("stations", [
    [], [(0.0, 0.0)], [(0, 0)], [(-0.0, 0.0), (0.0, -0.0)],
    [(0, 0), (0, 0), (0, 0)], [(1800, 0), (-1800, 0)],
    [(0, 1800), (0, -1800), (1800, 0), (-1800, 0)],
    [(2000000, -2000000), (-2000000, 2000000)],
    [(1e-13, 0), (0, 1e-13), (-1e-13, 0)],
    [(0, 0), (1e-10, 0), (2e-10, 0), (3e-10, 0)],
    [(0, 0), (1000, 1e-12), (-1000, -1e-12), (1700, 0)],
    [Position(0, 0), Position(500, 1000), Position(-500, 1000)],
])
def test_fixed_and_degenerate_sets(frozen_cover, stations):
    compare(frozen_cover, stations)


def test_random_finite_sets_and_duplicate_permutations(frozen_cover):
    rng = random.Random(861319)
    for _ in range(180):
        stations = [(rng.uniform(-2000, 2000), rng.uniform(-2000, 2000))
                    for _ in range(rng.randrange(1, 11))]
        radius = rng.choice([1e-12, 1.0, 1000.0, 1800.0, 2000000.0])
        expected = compare(frozen_cover, stations, radius)
        duplicate_permutation = stations + stations[::2]
        rng.shuffle(duplicate_permutation)
        assert_identical(compare(frozen_cover, duplicate_permutation, radius), expected)


@pytest.mark.parametrize("angle", [0.0, 1e-12, math.pi / 7, math.pi / 2, math.pi])
def test_collinear_and_ring_rotations(frozen_cover, angle):
    co, si = math.cos(angle), math.sin(angle)
    lines = [(-1600, 0), (-800, 0), (0, 0), (800, 0), (1600, 0)]
    ring = [(1200 * math.cos(i * math.tau / 6),
             1200 * math.sin(i * math.tau / 6)) for i in range(6)]
    for stations in (lines, ring):
        rotated = [(co * x - si * y, si * x + co * y) for x, y in stations]
        compare(frozen_cover, rotated)


@pytest.mark.parametrize("radius", [
    5e-324, 1e-200, 1e-150, 1e150, math.nextafter(1e150, math.inf),
    1e153, 1e154, 10**150, True,
])
def test_small_large_and_nonfast_arena_values(frozen_cover, radius):
    compare(frozen_cover, [(0.0, 0.0)], radius)


@pytest.mark.parametrize("radius", [1e150, 1e154])
def test_nonfinite_generated_witnesses_keep_old_nan_order(frozen_cover, radius):
    # rest / norm2 overflows to inf in the unchanged bisector construction;
    # its 0 * inf coordinates are NaN. The old ordered min/max ignores these
    # late NaNs, so treating NaN as infinity or rejecting it would differ.
    compare(frozen_cover, [(0.0, 0.0), (1e-8, 0.0)], radius)


def test_signed_zero_underflow_return(frozen_cover):
    result = compare(frozen_cover, [(-0.0, -0.0)], 5e-324)
    assert result.hex() == "0x0.0p+0"


@pytest.mark.parametrize("radius", [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan, "bad", None, 10**400])
def test_invalid_arena_exception_type_and_message(frozen_cover, radius):
    with pytest.raises(Exception) as old_error:
        frozen_cover.disk_cover_radius([(0, 0)], radius)
    with pytest.raises(type(old_error.value)) as new_error:
        disk_cover_radius([(0, 0)], radius)
    assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize("radius", [0, -1, math.nan, None, "bad", 10**400])
def test_empty_stations_keep_validation_order(frozen_cover, radius):
    # The original returns inf before validating the radius for an empty set.
    compare(frozen_cover, [], radius)


@pytest.mark.parametrize("stations", [
    None, [None], [(0,)], [(0, 1, 2)], [(True, 0)], [("0", 0)],
    [(math.nan, 0)], [(math.inf, 0)], [(2000000.0001, 0)],
])
def test_bad_stations_exception_type_and_message(frozen_cover, stations):
    with pytest.raises(Exception) as old_error:
        frozen_cover.disk_cover_radius(stations)
    with pytest.raises(type(old_error.value)) as new_error:
        disk_cover_radius(stations)
    assert str(new_error.value) == str(old_error.value)


@pytest.mark.parametrize("radius", [1e155, 1e200, 1e308])
def test_overflow_is_not_hidden_by_a_short_circuit(frozen_cover, radius):
    stations = [(0, 0), (1, 0)]
    with pytest.raises(OverflowError) as old_error:
        frozen_cover.disk_cover_radius(stations, radius)
    with pytest.raises(OverflowError) as new_error:
        disk_cover_radius(stations, radius)
    assert str(new_error.value) == str(old_error.value)


def test_custom_number_reduction_is_not_short_circuited(frozen_cover):
    class ObservedFloat(float):
        operations = []

        def __rsub__(self, value):
            self.operations.append(("rsub", float(value), float(self)))
            return float(value) - float(self)

    stations = [(ObservedFloat(-400), ObservedFloat(100)), (700.0, 0.0)]
    ObservedFloat.operations.clear()
    expected = frozen_cover.disk_cover_radius(stations)
    expected_operations = ObservedFloat.operations[:]
    ObservedFloat.operations.clear()
    actual = disk_cover_radius(stations)
    assert_identical(actual, expected)
    assert ObservedFloat.operations == expected_operations
    assert expected_operations


def test_generator_input_is_consumed_identically(frozen_cover):
    points = [(0, 0), (600, 0), (0, 600)]
    actual = disk_cover_radius((p for p in points))
    expected = frozen_cover.disk_cover_radius((p for p in points))
    assert_identical(actual, expected)


def test_certification_threshold_uses_identical_radius(frozen_cover):
    stations = [(1200 * math.cos(i * math.tau / 6),
                 1200 * math.sin(i * math.tau / 6)) for i in range(6)]
    radius = compare(frozen_cover, stations)
    for margin in [0.0, 1e-5, 0.1]:
        centre = radius + margin
        for reception in [math.nextafter(centre, -math.inf), centre,
                          math.nextafter(centre, math.inf)]:
            assert certifies_disk_cover(stations, reception, margin=margin) is (
                frozen_cover.certifies_disk_cover(stations, reception, margin=margin)
            )


def test_original_default_threshold_nearby_layouts(frozen_cover):
    # Bisection constructs a geometry near the real 1000 - 1e-5 certificate.
    # Only the immutable oracle determines the fixture, never any source truth.
    def layout(radius):
        return [(0.0, 0.0)] + [
            (radius * math.cos(i * math.tau / 6), radius * math.sin(i * math.tau / 6))
            for i in range(6)
        ]

    low, high = 800.0, 1500.0
    for _ in range(48):
        middle = (low + high) / 2
        if frozen_cover.certifies_disk_cover(layout(middle)):
            high = middle
        else:
            low = middle
    assert low < high
    for radius in [math.nextafter(low, -math.inf), low, high,
                   math.nextafter(high, math.inf)]:
        stations = layout(radius)
        compare(frozen_cover, stations)
        assert certifies_disk_cover(stations) is frozen_cover.certifies_disk_cover(stations)
    assert not frozen_cover.certifies_disk_cover(layout(low))
    assert frozen_cover.certifies_disk_cover(layout(high))
