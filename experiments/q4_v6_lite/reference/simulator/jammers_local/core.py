"""Simulation and protocol semantics; no account, upload or official service code.

Physical rules: SHA256 2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27.
HTTP business fields: supplied attachment 2. See COMPATIBILITY.md for limits.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import threading
import time
import unicodedata
from .algorithms import generate_practice, error_degrees, quantize_bearing, move_microseconds

PATHS = {'/enter', '/measure', '/clear', '/exit'}
MAX_BODY = 65536
BASE_FIELDS = {'arena_id', 'robot_id', 'request_id'}


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def identifier(value, limit):
    if not isinstance(value, str):
        return False
    try:
        return 1 <= len(value.encode('utf-8')) <= limit and all(
            unicodedata.category(c) not in {'Cc', 'Cf'} for c in value)
    except UnicodeEncodeError:
        return False


def integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


@dataclass(frozen=True)
class Source:
    channel: int
    x_um: int
    y_um: int
    max_receive_um: int
    kind: str
    direction_udeg: int | None

    @property
    def x(self): return self.x_um / 1_000_000
    @property
    def y(self): return self.y_um / 1_000_000


@dataclass(frozen=True)
class Scenario:
    problem_no: int
    sources: tuple[Source, ...]
    noise_seed: int
    seed_hex: str | None = None
    label: str = ''
    profile: str = 'practice'

    def __post_init__(self):
        if type(self.problem_no) is not int or self.problem_no not in (3, 4):
            raise ValueError('problem_no must be 3 or 4')
        if self.profile not in {'practice', 'fixture'}:
            raise ValueError('profile must be practice or fixture')
        low, high = (10, 16) if self.profile == 'practice' else (1, 20)
        if not low <= len(self.sources) <= high:
            raise ValueError(f'{self.profile} source count must be {low}..{high}')
        if not integer(self.noise_seed, 0, 2**64-1):
            raise ValueError('noise_seed must be uint64')
        if self.seed_hex is not None:
            try:
                if len(self.seed_hex) != 64 or len(bytes.fromhex(self.seed_hex)) != 32:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError('seed_hex must be 64 hex characters') from None
        channels = [s.channel for s in self.sources]
        if len(set(channels)) != len(channels) or channels != sorted(channels):
            raise ValueError('source channels must be unique and sorted')
        for s in self.sources:
            if not integer(s.channel, 1, 20):
                raise ValueError('invalid source channel')
            if type(s.x_um) is not int or type(s.y_um) is not int:
                raise ValueError('source positions must be integer micrometres')
            if s.x_um*s.x_um+s.y_um*s.y_um > 1_770_000_000**2:
                raise ValueError('source is outside the 1770 m disk')
            if not integer(s.max_receive_um, 1_000_000_000, 1_500_000_000):
                raise ValueError('invalid receive radius')
            if s.kind == 'omni':
                if s.direction_udeg is not None:
                    raise ValueError('omni source cannot have a direction')
            elif s.kind == 'directional':
                if not integer(s.direction_udeg, 0, 359_999_999):
                    raise ValueError('invalid directional orientation')
            else:
                raise ValueError('unknown source kind')
        k = sum(s.kind == 'directional' for s in self.sources)
        if (self.problem_no == 3 and k) or (self.problem_no == 4 and not k):
            raise ValueError('source types do not match the problem number')

    @classmethod
    def generate(cls, problem=4, seed='demo-20260912', *, seed_hex=None):
        # A text seed is a LOCAL convenience; the official generator input is 32 bytes.
        raw = bytes.fromhex(seed_hex) if seed_hex is not None else hashlib.sha256(str(seed).encode()).digest()
        data = generate_practice(raw, problem)
        return cls(problem, tuple(Source(**s) for s in data['sources']), data['noise_seed'], raw.hex(), str(seed))

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 'local-jammers-v1':
            raise ValueError('expected local-jammers-v1 scenario format')
        return cls(data['problem_no'], tuple(Source(**s) for s in data['sources']),
                   data['noise_seed'], data.get('seed_hex'), data.get('label', ''), data.get('profile', 'practice'))

    def as_dict(self):
        from dataclasses import asdict
        return {'schema_version': 'local-jammers-v1', **asdict(self)}

    @property
    def case_id(self):
        data = self.as_dict()
        data.pop('label')
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class Engine:
    def __init__(self, scenario, *, max_virtual_us=360_000_000_000):
        if not integer(max_virtual_us, 1, 2**63-1):
            raise ValueError('max_virtual_us must be a positive int64')
        self.scenario = scenario
        self.sources = {s.channel: s for s in scenario.sources}
        self.position = (0.0, 0.0)
        self.channel = 1
        self.cleared = set()
        self.time_us = 0
        self.max_virtual_us = max_virtual_us
        self.lifecycle = 'ready'
        self.stop_reason = None
        self.costs_us = dict(movement=0, switching=0, detection=0, optical=0, removal=0)
        self.distance_m = 0.0
        self.measurements = self.clear_attempts = self.failed_clears = self.switches = 0

    def finish(self, reason):
        self.lifecycle, self.stop_reason = 'ended', reason

    @staticmethod
    def covered(source, x, y, distance):
        if source.kind == 'omni' or distance == 0:
            return True
        theta = (math.atan2(y-source.y, x-source.x)*180/math.pi) % 360
        return abs(math.remainder(theta-source.direction_udeg/1_000_000, 360)) <= 90.000000001

    def apply(self, path, payload):
        """Validated actions only. A None result is a non-mutating business rejection."""
        if path not in PATHS or self.lifecycle == 'ended': return None
        if path == '/enter':
            if self.lifecycle != 'ready': return None
            self.lifecycle = 'active'
            return {}
        if self.lifecycle != 'active': return None
        if path == '/exit':
            self.finish('user_exit')
            return {'exit_reason': 'user_exit'}
        x, y = payload['position']['x'], payload['position']['y']
        c = payload['channel']
        if not all(number(v) and abs(v) <= 2_000_000 for v in (x, y)) or not integer(c, 1, 20):
            return None
        x, y = float(x), float(y)
        travel = move_microseconds(*self.position, x, y)
        self.distance_m += math.hypot(x-self.position[0], y-self.position[1])
        self.costs_us['movement'] += travel
        self.position = (x, y)
        source = self.sources.get(c) if c not in self.cleared else None
        distance = math.hypot(source.x-x, source.y-y) if source else math.inf
        if path == '/measure':
            switch = int(c != self.channel)
            self.costs_us['switching'] += switch*1_000_000
            self.costs_us['detection'] += 5_000_000
            self.switches += switch
            self.channel = c
            self.measurements += 1
            if source is None or distance > source.max_receive_um/1_000_000 or not self.covered(source,x,y,distance):
                detail = {'measure_result': 'no_signal'}
            elif distance <= 5:
                detail = {'measure_result': 'near'}
            else:
                theta = (math.atan2(source.y-y,source.x-x)*180/math.pi) % 360
                error = error_degrees(self.scenario.noise_seed,c,x,y)
                detail = {'measure_result': 'direction', 'svd_deg': quantize_bearing(theta,error)}
        else:
            success = source is not None and distance <= 20
            self.clear_attempts += 1
            self.costs_us['optical'] += 3_000_000
            self.costs_us['removal'] += int(success)*2_000_000
            if success: self.cleared.add(c)
            else: self.failed_clears += 1
            detail = {'clear_result': 'success' if success else 'no_target_in_range'}
        self.time_us = sum(self.costs_us.values())
        if self.time_us >= self.max_virtual_us: self.finish('virtual_timeout')
        return detail

    def snapshot(self, truth=False):
        result = dict(lifecycle=self.lifecycle, stop_reason=self.stop_reason,
                      position=dict(x=self.position[0],y=self.position[1]), current_channel=self.channel,
                      virtual_time_s=self.time_us/1_000_000, virtual_time_us=self.time_us,
                      cleared_channels=sorted(self.cleared), cleared_count=len(self.cleared),
                      distance_m=self.distance_m, measurement_count=self.measurements,
                      clear_attempt_count=self.clear_attempts, failed_clear_count=self.failed_clears,
                      switch_count=self.switches, time_breakdown_s={k:v/1_000_000 for k,v in self.costs_us.items()})
        if truth:
            result.update(source_total=len(self.sources), all_cleared=len(self.cleared)==len(self.sources),
                          scenario=self.scenario.as_dict())
        return result


def validate(path, payload, robot_id):
    if path not in PATHS: return 404, None
    if not isinstance(payload, dict): return 400, None
    expected = BASE_FIELDS | ({'position','channel'} if path in {'/measure','/clear'} else set())
    if not expected <= payload.keys(): return 400, None
    if not isinstance(payload['arena_id'],str) or not identifier(payload['robot_id'],64) or not identifier(payload['request_id'],128):
        return 400, None
    normalized = {k:payload[k] for k in BASE_FIELDS}
    unknown = set(payload) != expected
    if 'position' in expected:
        p, c = payload['position'], payload['channel']
        if not isinstance(p,dict) or not {'x','y'} <= p.keys(): return 400, None
        if not all(number(p[a]) and abs(p[a]) <= 2_000_000 for a in ('x','y')): return 400, None
        if not number(c) or not 1 <= c <= 20 or c != int(c): return 400, None
        unknown |= set(p) != {'x','y'}
        normalized.update(position={a:float(p[a]) if p[a] else 0.0 for a in ('x','y')},channel=int(c))
    if unknown or payload['arena_id'] != 'default' or payload['robot_id'] != robot_id: return 200, None
    return 200, normalized


class Session:
    def __init__(self, scenario, *, robot_id='local-test', clock=time.monotonic,
                 wall_clock=time.time, max_real_s=1200, window_s=1500,
                 max_virtual_us=360_000_000_000, max_entries=131072):
        if not identifier(robot_id,64): raise ValueError('invalid robot_id')
        if not all(number(v) and v>0 for v in (max_real_s,window_s)): raise ValueError('time limits must be positive')
        self.engine = Engine(scenario,max_virtual_us=max_virtual_us)
        self.robot_id, self.clock, self.wall_clock = robot_id, clock, wall_clock
        self.created = clock()
        self.started = None
        self.window_s, self.max_real_s = float(window_s), float(max_real_s)
        self.max_entries = max_entries
        self.cache = {}
        self.history = []
        self._lock = threading.RLock()
        self._inflight = None

    def rejection(self):
        return dict(accepted=False, real_timestamp_ms=int(self.wall_clock()*1000), virtual_time_s=0)

    def _expired(self, now):
        if self.engine.lifecycle == 'ended': return True
        if now >= self.created+self.window_s:
            self.engine.finish('window_timeout'); return True
        if self.started is not None and now >= self.started+self.max_real_s:
            self.engine.finish('program_timeout'); return True
        return False

    def dispatch(self,path,payload):
        status, normalized = validate(path,payload,self.robot_id)
        if normalized is None: return status,self.rejection()
        rid = normalized['request_id']
        canonical = json.dumps([path,normalized],sort_keys=True,separators=(',',':'),ensure_ascii=False)
        while True:
            with self._lock:
                if rid in self.cache:
                    old_key, response = self.cache[rid]
                    return (200,deepcopy(response)) if old_key==canonical else (409,self.rejection())
                if self._inflight is not None:
                    pending_key,event = self._inflight
                    if pending_key != canonical: return 409,self.rejection()
                else:
                    if self._expired(self.clock()): return 200,self.rejection()
                    if len(self.cache)>=self.max_entries: return 429,self.rejection()
                    event=threading.Event()
                    self._inflight=(canonical,event)
                    break
            event.wait()
        # Exactly one registered action may mutate the engine. Snapshot waits for it.
        try:
            detail=self.engine.apply(path,normalized)
            if detail is None: return 200,self.rejection()
            if path=='/enter':
                self.started=self.clock()
                remaining=max(0,math.floor(min(self.max_real_s,self.created+self.window_s-self.started)))
                detail.update(max_virtual_duration_s=self.engine.max_virtual_us/1_000_000,
                              max_real_duration_s=self.max_real_s,remaining_real_duration_s=remaining)
            response=dict(accepted=True,real_timestamp_ms=int(self.wall_clock()*1000),
                          virtual_time_s=self.engine.time_us/1_000_000,**detail)
            with self._lock:
                self.cache[rid]=(canonical,deepcopy(response))
                self.history.append(dict(index=len(self.history)+1,path=path,request=deepcopy(normalized),response=deepcopy(response)))
            return 200,response
        finally:
            with self._lock:
                self._inflight=None
                event.set()

    def snapshot(self,truth=False):
        while True:
            with self._lock:
                if self._inflight is None:
                    self._expired(self.clock())
                    return dict(case_id=self.engine.scenario.case_id,problem_no=self.engine.scenario.problem_no,
                                profile=self.engine.scenario.profile,robot_id=self.robot_id,
                                action_count=len(self.history),**self.engine.snapshot(truth))
                event=self._inflight[1]
            event.wait()

    def export(self):
        while True:
            with self._lock:
                if self._inflight is None:
                    state=self.snapshot(True)
                    return dict(schema_version='local-jammers-run-v1',provenance='static-reconstruction-not-official',
                                limits=dict(max_virtual_us=self.engine.max_virtual_us,max_real_s=self.max_real_s,window_s=self.window_s),
                                state=state,history=deepcopy(self.history))
                event=self._inflight[1]
            event.wait()


class LocalClient:
    """Observation-only policy surface. The Python process is not a security sandbox."""
    def __init__(self,dispatch,robot_id='local-test',*,cancel=None):
        self._dispatch,self.robot_id,self._cancel=dispatch,robot_id,cancel
        self.sequence=0
        self.position=(0.,0.)
        self.cleared=set()
        self.virtual_time_s=0.
        self.active=False

    def _send(self,path,**fields):
        if self._cancel is not None and self._cancel.is_set(): raise RuntimeError('run cancelled')
        self.sequence+=1
        payload=dict(arena_id='default',robot_id=self.robot_id,request_id=f'policy-{self.sequence}',**fields)
        status,response=self._dispatch(path,payload)
        if status!=200 or not response['accepted']: raise RuntimeError(f'{path} rejected (HTTP {status})')
        self.virtual_time_s=response['virtual_time_s']
        if 'position' in fields: self.position=(fields['position']['x'],fields['position']['y'])
        if response.get('clear_result')=='success': self.cleared.add(fields['channel'])
        if path=='/enter': self.active=True
        if path=='/exit': self.active=False
        return response

    def enter(self): return self._send('/enter')
    def exit(self): return self._send('/exit')
    def measure(self,position,channel): return self._send('/measure',position=dict(x=position[0],y=position[1]),channel=channel)
    def clear(self,position,channel): return self._send('/clear',position=dict(x=position[0],y=position[1]),channel=channel)
