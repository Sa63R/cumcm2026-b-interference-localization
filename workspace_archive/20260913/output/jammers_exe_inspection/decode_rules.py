"""Reproduce rule extraction from this exact executable, without executing it."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / 'inventory.json').read_text())
B = Path(META['source']).read_bytes()
EXPECTED = '2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27'
assert hashlib.sha256(B).hexdigest() == EXPECTED, 'Different exe; addresses must be rediscovered.'

def offset(address, length=1):
    for s in META['sections']:
        start = META['image_base'] + s['rva']
        if start <= address and address + length <= start + s['raw_size']:
            return s['raw_offset'] + address - start
    raise ValueError(hex(address))

def type_name(address):
    p = offset(address) + 1
    n = shift = 0
    while True:
        c = B[p]
        p += 1
        n |= (c & 127) << shift
        if c < 128:
            return B[p:p + n].decode()
        shift += 7

def fields(typ):
    ptr, n, cap = struct.unpack_from('<3Q', B, offset(typ) + 56)
    assert n == cap and n < 100
    return [struct.unpack_from('<3Q', B, offset(ptr) + 24 * j) for j in range(n)]

def read_value(position, typ):
    kind = B[offset(typ) + 23] & 31
    if kind == 24:
        ptr, n = struct.unpack_from('<2Q', B, position)
        return B[offset(ptr):offset(ptr) + n].decode()
    if kind == 6:
        return struct.unpack_from('<q', B, position)[0]
    if kind == 8:
        return B[position]
    raise ValueError(kind)

result = {}
for label, typ, address in [('generation_rules', 0x14107c010, 0x140907530),
                            ('simulation_rules', 0x14107eee0, 0x140907ea0)]:
    f = fields(typ)
    result[label] = dict(value_address=hex(address), type_address=hex(typ),
                        values={type_name(n): read_value(offset(address) + pos, t) for n, t, pos in f},
                        fields=[dict(name=type_name(n), byte_offset=pos, type_address=hex(t))
                                for n, t, pos in f])
(ROOT / 'decoded_rule_structs.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v['values'] for k, v in result.items()}, ensure_ascii=False, indent=2))
