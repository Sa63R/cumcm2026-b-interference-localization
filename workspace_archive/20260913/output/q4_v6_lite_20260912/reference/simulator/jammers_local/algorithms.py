"""Readable reconstruction of selected algorithms from static assembly.

These functions are illustrative reconstructions, not original Go source.
They have not been checked against a running official executable. They use
explicit synthetic seeds and do not access accounts, sessions or servers.
"""
import hashlib
import hmac
import math
from collections import defaultdict

def round_away(x):
    # Avoid the extra rounding introduced by computing x + 0.5 first.
    fractional, integral = math.modf(x)
    return int(integral) + (1 if fractional >= .5 else -1 if fractional <= -.5 else 0)

class CounterSource:
    def __init__(self, seed):
        if len(seed) != 32:
            raise ValueError('Generator seed must be 32 bytes.')
        self.seed = seed
        self.counters = defaultdict(int)

    def next(self, label):
        counter = self.counters[label]
        payload = b'practice-case-v1\0' + label.encode() + b'\0' + counter.to_bytes(8, 'big')
        self.counters[label] += 1
        digest = hmac.new(self.seed, payload, hashlib.sha256).digest()
        return int.from_bytes(digest[:8], 'big')

    def uint_n(self, label, n):
        threshold = (1 << 64) % n
        while True:
            x = self.next(label)
            if x >= threshold:
                return x % n

    def shuffle(self, label, values):
        # Reverse Fisher-Yates, matching the inspected loop.
        for i in range(len(values) - 1, 0, -1):
            j = self.uint_n(label, i + 1)
            values[i], values[j] = values[j], values[i]

def generate_practice(seed, problem):
    if problem not in (3, 4):
        raise ValueError(problem)
    r = CounterSource(seed)
    count = 10 + r.uint_n('count', 7)
    channels = list(range(1, 21))
    r.shuffle('channels', channels)
    channels = sorted(channels[:count])
    directional = set()
    if problem == 4:
        k = 1 + r.uint_n('directional-count', count)
        order = channels.copy()
        r.shuffle('directional-channels', order)
        directional = set(order[:k])
    sources = []
    for channel in channels:
        while True:
            u = (r.next(f'jammer/{channel}/radius') >> 11) * 2**-53
            v = (r.next(f'jammer/{channel}/theta') >> 11) * 2**-53
            radius = 1_770_000_000 * math.sqrt(u)
            theta = 2 * math.pi * v
            x = round_away(radius * math.cos(theta))
            y = round_away(radius * math.sin(theta))
            if x*x + y*y <= 1_770_000_000**2:
                break
        receive = 1_000_000_000 + r.uint_n(f'jammer/{channel}/receive', 500_000_001)
        direction = r.uint_n(f'jammer/{channel}/direction', 360_000_000) if channel in directional else None
        sources.append(dict(channel=channel, x_um=x, y_um=y, max_receive_um=receive,
                            kind='directional' if direction is not None else 'omni',
                            direction_udeg=direction))
    return dict(problem_no=problem, sources=sources, noise_seed=r.next('noise-seed'))

def grid(noise_seed, channel, i, j):
    payload = f'{noise_seed}:{channel}:{i}:{j}'.encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    value = int.from_bytes(digest, 'big')
    return 2 * (float(value) / 2**64) - 1

def error_degrees(noise_seed, channel, x, y):
    u, v = x/150, y/150
    i, j = math.floor(u), math.floor(v)
    u, v = min(1, max(0, u-i)), min(1, max(0, v-j))
    a, b = u*u*(3-2*u), v*v*(3-2*v)
    g00 = grid(noise_seed, channel, i, j)
    g10 = grid(noise_seed, channel, i+1, j)
    g01 = grid(noise_seed, channel, i, j+1)
    g11 = grid(noise_seed, channel, i+1, j+1)
    low, high = g00 + (g10-g00)*a, g01 + (g11-g01)*a
    return low + (high-low)*b

def quantize_bearing(theta, error):
    raw = round_away((theta+error)*100)
    low = math.ceil((theta-1)*100)
    high = math.floor((theta+1)*100)
    return min(high, max(low, raw)) % 36000 / 100

def move_microseconds(x0, y0, x1, y1):
    return round_away(math.hypot(x1-x0, y1-y0) * 1e12 / 5_000_000)
