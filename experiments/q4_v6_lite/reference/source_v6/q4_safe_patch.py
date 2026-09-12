"""Low-risk patch for the unchanged V2 strategy.

After 16 DIFFERENT source channels have produced a positive observation, every
other channel is provably empty under the problem's count bound. Suppress those
requests while retaining V2's physical route and known-target measurements.
This is logical inference, not a fabricated measurement or a server call.
"""
from __future__ import annotations
import q4_baseline as b
from q4_fast_solver import solve_fast,FastConfig

class CountBoundDevice:
    def __init__(self,device:b.Device):
        self._device=device
        self.found:set[int]=set()
        self.skipped_empty_detections=0
    @property
    def position(self):return self._device.position
    @property
    def channel(self):return self._device.channel
    def move(self,p):self._device.move(p)
    def detect(self,channel):
        if len(self.found)==16 and channel not in self.found:
            self.skipped_empty_detections+=1
            return b.Observation('none')
        obs=self._device.detect(channel)
        if obs.status!='none':self.found.add(channel)
        if len(self.found)>16:raise RuntimeError('Source-count assumption violated')
        return obs
    def clear(self,channel):return self._device.clear(channel)

def solve_safe(device:b.Device,config:FastConfig|None=None):
    proxy=CountBoundDevice(device)
    report=solve_fast(proxy,config or FastConfig())
    report['skipped_provably_empty_detections']=proxy.skipped_empty_detections
    return report
