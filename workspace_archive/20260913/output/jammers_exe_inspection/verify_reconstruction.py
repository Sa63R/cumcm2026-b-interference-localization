"""Property checks of our reconstruction; not official-executable parity tests."""
from pathlib import Path
import hashlib
import json
import math
import random
from collections import Counter
from reconstructed_rules import generate_practice, error_degrees, quantize_bearing, move_microseconds

ROOT = Path(__file__).resolve().parent
counts = Counter()
all_directional = 0
source_count = 0
for i in range(2000):
    seed = hashlib.sha256(f'explanation-synthetic-seed-{i}'.encode()).digest()
    a = generate_practice(seed, 3)
    b = generate_practice(seed, 4)
    assert b == generate_practice(seed, 4)
    assert a['noise_seed'] == b['noise_seed']
    sources = b['sources']
    n = len(sources)
    counts[n] += 1
    assert 10 <= n <= 16
    channels = [s['channel'] for s in sources]
    assert len(set(channels)) == n and channels == sorted(channels)
    assert min(channels) >= 1 and max(channels) <= 20
    assert all(s['kind'] == 'omni' for s in a['sources'])
    k = sum(s['kind'] == 'directional' for s in sources)
    assert 1 <= k <= n
    all_directional += k == n
    for q3, q4 in zip(a['sources'], sources):
        for field in ['channel', 'x_um', 'y_um', 'max_receive_um']:
            assert q3[field] == q4[field]
        assert q4['x_um']**2 + q4['y_um']**2 <= 1_770_000_000**2
        assert 1_000_000_000 <= q4['max_receive_um'] <= 1_500_000_000
        if q4['kind'] == 'directional':
            assert 0 <= q4['direction_udeg'] < 360_000_000
        else:
            assert q4['direction_udeg'] is None
    source_count += n

rng = random.Random(20260912)
max_error = 0
max_quantized_error = 0
for i in range(10000):
    x, y = rng.uniform(-2000, 2000), rng.uniform(-2000, 2000)
    seed, c = rng.getrandbits(64), rng.randint(1, 20)
    error = error_degrees(seed, c, x, y)
    assert error == error_degrees(seed, c, x, y)
    assert -1.000000000000001 <= error <= 1.000000000000001
    theta = rng.uniform(0, 360)
    angle = quantize_bearing(theta, error)
    wrapped_error = abs((angle-theta+180) % 360-180)
    assert 0 <= angle < 360 and wrapped_error <= 1+1e-10
    max_error = max(max_error, abs(error))
    max_quantized_error = max(max_quantized_error, wrapped_error)
    dx, dy = rng.uniform(-2, 2), rng.uniform(-2, 2)
    nearby = error_degrees(seed, c, x+dx, y+dy)
    assert abs(nearby-error) <= .02*(abs(dx)+abs(dy))+1e-12

assert quantize_bearing(12.346, .9999) == 13.34
assert move_microseconds(0, 0, 3, 4) == 1_000_000
assert move_microseconds(0, 0, 0, 0) == 0
assert move_microseconds(0, 0, 100, 0) == 20_000_000
result = dict(scope='Property checks on Python reconstruction only; official exe was not run.',
              synthetic_seed_count=2000, generated_case_count=4000,
              sources_in_problem4_cases=source_count,
              jammer_count_histogram=dict(sorted(counts.items())),
              all_directional_problem4_cases=all_directional,
              noise_sample_count=10000,
              max_abs_noise_observed=max_error,
              max_abs_quantized_error_observed=max_quantized_error,
              invariants='PASS', outer_ring_area_fraction=1-(1770/1800)**2)
(ROOT/'reconstruction_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
