"""Device adapter for the local reconstructed simulator, copied from the frozen V6 benchmark."""
import math
import q4_baseline as b

class Device:
    """Only accepted measure/clear feedback; move is combined with next action."""
    __slots__ = ('__client', '_position', '_channel')
    def __init__(self, client):
        self.__client = client
        self._position, self._channel = (0.,0.), 1
    @property
    def position(self): return self._position
    @property
    def channel(self): return self._channel
    def move(self, point):
        if len(point) != 2 or not all(math.isfinite(v) and abs(v)<=2_000_000 for v in point):
            raise ValueError('Invalid destination')
        self._position = tuple(map(float, point))
    def detect(self, channel):
        response = self.__client.measure(self.position, channel)
        self._channel = channel
        kind = response['measure_result']
        if kind == 'direction': return b.Observation('bearing', math.radians(response['svd_deg']))
        if kind == 'near': return b.Observation('strong')
        if kind == 'no_signal': return b.Observation('none')
        raise RuntimeError(f'Unexpected measure response: {kind}')
    def clear(self, channel):
        response = self.__client.clear(self.position, channel)
        if response['clear_result'] not in ('success', 'no_target_in_range'):
            raise RuntimeError('Unexpected clear response')
        return response['clear_result'] == 'success'
