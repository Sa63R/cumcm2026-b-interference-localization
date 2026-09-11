"""Exact circle memo checks; no new simulation cases or model changes."""
from dataclasses import FrozenInstanceError
import math

import pytest

from geometry import disk_halfplanes
import localization
from localization import CandidateRegion, _cached_reception_disk, _reception_disk_halfplanes
from localization.omni import OmniCandidateRegion


@pytest.fixture(autouse=True)
def empty_cache():
    _cached_reception_disk.cache_clear()
    yield
    _cached_reception_disk.cache_clear()


def hex_planes(values):
    return tuple(tuple(float(getattr(hp, field)).hex() for field in ("a", "b", "c")) for hp in values)


@pytest.mark.parametrize("center", [(0., 0.), (-0., 0.), (1800., -1800.), (1e-300, -1e-300), (123.456, -789.012)])
@pytest.mark.parametrize("radius,sides", [(1500., 128), (1000, 64), (19.9, 8)])
def test_original_coefficients_are_bit_exact_and_cached(center, radius, sides):
    expected = disk_halfplanes(center, radius, sides, outer=True)
    first = _reception_disk_halfplanes(center, radius, sides)
    second = _reception_disk_halfplanes(center, radius, sides)
    assert hex_planes(expected) == hex_planes(first) == hex_planes(second)
    assert first is second
    assert _cached_reception_disk.cache_info().hits == 1
    with pytest.raises(FrozenInstanceError):
        second[0].c = 0.


def test_exact_keys_do_not_merge_signed_zero_or_nearby_inputs():
    for center, radius, sides in [((0., 0.), 1500., 128), ((-0., 0.), 1500., 128),
                                  ((0., -0.), 1500., 128), ((math.nextafter(0., 1.), 0.), 1500., 128),
                                  ((0., 0.), math.nextafter(1500., math.inf), 128), ((0., 0.), 1500., 64)]:
        _reception_disk_halfplanes(center, radius, sides)
    assert _cached_reception_disk.cache_info().misses == 6


@pytest.mark.parametrize("center,radius,sides", [
    ((0., 0.), 0., 128), ((0., 0.), -1., 128), ((0., 0.), math.nan, 128),
    ((0., 0.), math.inf, 128), ((math.nan, 0.), 1500., 128), ((math.inf, 0.), 1500., 128),
    ((0., 0.), 1500., 7), ((0., 0.), 1500., True), ((0., 0.), 1500., 8.0),
    ((0.,), 1500., 128), ([0., 0.], "1500", 128), ((0., 0.), 10 ** 1000, 128),
])
def test_invalid_arguments_preserve_original_exception(center, radius, sides):
    errors = []
    for function in (lambda c, r, s: disk_halfplanes(c, r, s, outer=True), _reception_disk_halfplanes):
        with pytest.raises(Exception) as caught:
            function(center, radius, sides)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]
    assert _cached_reception_disk.cache_info().currsize == 0


class UnhashableFloat(float):
    def __hash__(self):
        raise AssertionError("Unusual radius should not be hashed")


class UnhashableInt(int):
    def __hash__(self):
        raise AssertionError("Unusual sides should not be hashed")


@pytest.mark.parametrize("center,radius,sides", [
    ([0., 0.], 1500., 128), ((0, 0), 1500., 128), ((0., 0.), True, 8),
    ((0., 0.), UnhashableFloat(1500), 128), ((0., 0.), 1500., UnhashableInt(128)),
])
def test_unusual_types_use_original_uncached_path(center, radius, sides):
    assert hex_planes(_reception_disk_halfplanes(center, radius, sides)) == hex_planes(disk_halfplanes(center, radius, sides, outer=True))
    assert _cached_reception_disk.cache_info().currsize == 0


def test_entry_count_is_bounded():
    for index in range(135):
        _reception_disk_halfplanes((float(index), 0.), 1500., 8)
    assert _cached_reception_disk.cache_info().currsize == 128
    old_misses = _cached_reception_disk.cache_info().misses
    _reception_disk_halfplanes((0., 0.), 1500., 8)
    assert _cached_reception_disk.cache_info().misses == old_misses + 1


def observe_sequence(cls):
    region = cls()
    sequence = [("negative", (1600., 300.), None), ("positive", (0., 0.), 0.),
                ("positive", (700., -500.), math.degrees(math.atan2(500., 100.))),
                ("negative", (1600., -300.), None), ("positive", (700., -500.), 78.69)]
    snapshots = []
    for kind, p, bearing in sequence:
        if kind == "negative":
            if isinstance(region, OmniCandidateRegion):
                region.observe_no_signal(p)
        else:
            region.observe(p, bearing)
        circle = region.enclosing_disk() if region.vertices else None
        snapshots.append((tuple(tuple(float(v).hex() for v in xy) for xy in region.vertices),
            None if circle is None else (tuple(float(v).hex() for v in circle.center), circle.radius.hex())))
        clone = region.copy()
        assert clone.vertices is region.vertices
        assert clone.observations is not region.observations
    return snapshots


@pytest.mark.parametrize("cls", [CandidateRegion, OmniCandidateRegion])
def test_full_observation_clip_order_and_mec_match_original(cls, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(localization, "_reception_disk_halfplanes", lambda c, r, s: disk_halfplanes(c, r, s, outer=True))
        expected = observe_sequence(cls)
    actual = observe_sequence(cls)
    assert actual == expected
    assert _cached_reception_disk.cache_info().hits >= 1


def test_hypothetical_copies_share_only_circle_constraints():
    region = OmniCandidateRegion().observe((0., 0.), 0.)
    original_vertices, original_observations = region.vertices, tuple(region.observations)
    first, second = region.copy(), region.copy()
    first.observe((700., -500.), 78.6)
    misses = _cached_reception_disk.cache_info().misses
    second.observe((700., -500.), 78.8)
    assert _cached_reception_disk.cache_info().misses == misses
    assert region.vertices == original_vertices and tuple(region.observations) == original_observations
    assert first.vertices != second.vertices
    assert first.observations is not second.observations
