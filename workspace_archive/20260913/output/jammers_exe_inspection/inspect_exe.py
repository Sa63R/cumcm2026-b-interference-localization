"""Read a PE/Go binary as data. Never load or execute its code."""
from pathlib import Path
import collections
import hashlib
import json
import re
import struct
import subprocess

SOURCE = Path('/Users/zephyrr/Downloads/CUMCM2026B/Jammers-simulator/jammers-simulator.exe')
OUT = Path(__file__).resolve().parent
B = SOURCE.read_bytes()
PE = struct.unpack_from('<I', B, 60)[0]
COUNT = struct.unpack_from('<H', B, PE + 6)[0]
OPT_SIZE = struct.unpack_from('<H', B, PE + 20)[0]
OPT = PE + 24
BASE = struct.unpack_from('<Q', B, OPT + 24)[0]
SECTIONS = []
for i in range(COUNT):
    off = OPT + OPT_SIZE + i * 40
    name, vsize, rva, size, raw = struct.unpack_from('<8sIIII', B, off)
    SECTIONS.append(dict(name=name.rstrip(b'\0').decode(), virtual_size=vsize,
                         rva=rva, raw_size=size, raw_offset=raw))

def offset(addr, size=1):
    for s in SECTIONS:
        start = BASE + s['rva']
        if start <= addr and addr + size <= start + s['raw_size']:
            return s['raw_offset'] + addr - start
    return None

def cstring(off):
    end = B.find(b'\0', off)
    return B[off:end].decode('utf-8', 'replace')

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

resources = []
json_literals = {}
for section in SECTIONS:
    if section['name'] not in ('.rdata', '.data'):
        continue
    start, size = section['raw_offset'], section['raw_size']
    for i in range(start, start + size - 48, 8):
        ptr, length = struct.unpack_from('<QQ', B, i)
        if not 1 <= length <= 200000:
            continue
        pos = offset(ptr, length)
        if pos is None:
            continue
        if 80 <= length <= 30000 and B[pos:pos + 2] == b'{"':
            try:
                value = json.loads(B[pos:pos + length])
            except (ValueError, UnicodeError):
                pass
            else:
                json_literals[str(pos)] = value
        if length > 200:
            continue
        name_bytes = B[pos:pos + length]
        if not re.fullmatch(rb'(?:frontend/dist|testdata|assets)/[A-Za-z0-9_./-]+', name_bytes):
            continue
        data_ptr, data_len = struct.unpack_from('<QQ', B, i + 16)
        if data_len > 2000000:
            continue
        data_off = offset(data_ptr, data_len)
        if data_off is None:
            continue
        name = name_bytes.decode()
        target = OUT / 'embedded' / name
        if '..' in Path(name).parts:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = B[data_off:data_off + data_len]
        target.write_bytes(payload)
        resources.append(dict(name=name, size=data_len, file_offset=data_off,
                              descriptor_offset=i, sha256=hashlib.sha256(payload).hexdigest()))

# Go's function-name and PC tables remain useful even with a stripped COFF table.
functions = []
magic = b'\xf1\xff\xff\xff\x00\x00\x01\x08'
for match in re.finditer(re.escape(magic), B):
    pcln = match.start()
    nfunc, nfiles, text_start, funcname, cutab, filetab, pctab, table = struct.unpack_from('<8Q', B, pcln + 8)
    if not (1 <= nfunc < 100000 and 0 < table < len(B) - pcln):
        continue
    table_off = pcln + table
    # This PE stores zero in pcHeader.textStart; PCs are relative to .text.
    runtime_text_start = text_start or (BASE + next(s['rva'] for s in SECTIONS if s['name'] == '.text'))
    for j in range(nfunc):
        entry, foff = struct.unpack_from('<II', B, table_off + j * 8)
        next_entry = struct.unpack_from('<I', B, table_off + (j + 1) * 8)[0]
        fdata = table_off + foff
        name_off = struct.unpack_from('<i', B, fdata + 4)[0]
        name = cstring(pcln + funcname + name_off)
        functions.append(dict(name=name, address=runtime_text_start + entry,
                              end_address=runtime_text_start + next_entry,
                              size=next_entry - entry))
    break

imports = []
import_rva, import_size = struct.unpack_from('<II', B, OPT + 112 + 8)
imp = offset(BASE + import_rva)
if imp is not None:
    while any(B[imp:imp + 20]):
        original, stamp, chain, name_rva, thunk = struct.unpack_from('<5I', B, imp)
        dll = cstring(offset(BASE + name_rva))
        names = []
        p = offset(BASE + (original or thunk))
        while p is not None:
            val = struct.unpack_from('<Q', B, p)[0]
            if not val:
                break
            if val & (1 << 63):
                names.append('ordinal:' + str(val & 65535))
            else:
                names.append(cstring(offset(BASE + val) + 2))
            p += 8
        imports.append(dict(dll=dll, functions=names))
        imp += 20

build = B.find(b'\xff Go buildinf:')
def varint(pos):
    result = shift = 0
    while True:
        c = B[pos]
        pos += 1
        result |= (c & 127) << shift
        if c < 128:
            return result, pos
        shift += 7

compiler = None
build_text = ''
if build >= 0 and B[build + 15] & 2:
    n, p = varint(build + 32)
    compiler = B[p:p + n].decode()
    n, p = varint(p + n)
    build_text = B[p:p + n].decode('utf-8', 'replace')
    first = build_text.find('path\t')
    last = build_text.rfind('\n')
    if first >= 0 and last > first:
        build_text = build_text[first:last + 1]
    (OUT / 'go_build_info.txt').write_text(build_text)

source_paths = sorted(set(x.decode() for x in re.findall(rb'jammers/(?:client|common)/[A-Za-z0-9_./-]+\.go', B)))
app_functions = [f for f in functions if f['name'].startswith('jammers/')]
module_counts = collections.Counter(f['name'].split('/internal/')[-1].split('/')[0].split('.')[0] for f in app_functions)
summary = dict(source=str(SOURCE), size=len(B), sha256=hashlib.sha256(B).hexdigest(),
               machine='Windows AMD64', image_base=BASE, compiler=compiler,
               certificate_directory=struct.unpack_from('<II', B, OPT + 112 + 4 * 8),
               sections=SECTIONS, embedded_resources=resources,
               function_count=len(functions), app_function_count=len(app_functions),
               app_module_function_counts=dict(sorted(module_counts.items())),
               source_paths=source_paths, direct_imports=imports)
save_json('inventory.json', summary)
save_json('function_index.json', app_functions)
save_json('all_function_index.json', functions)
save_json('json_literals.json', json_literals)
(OUT / 'function_names.txt').write_text('\n'.join(f['name'] for f in app_functions) + '\n')

# Produce a few narrowly selected disassemblies; objdump also reads the file as data.
by_address = {f['address']: f['name'] for f in functions}
selected = [f for f in functions if any(tag in f['name'] for tag in
            ['/bearingnoise.', '/scenario.', '/simcore.',
             '/runcontroller.(*Controller).StartPractice',
             '/runcontroller.(*Controller).StartFormal',
             '/robotapi.requiredCoordinate', '/robotapi.requiredChannel',
             '/robotapi.canonicalRequest', '/robotapi.virtualSeconds',
             '/testsession.(*Session).Execute', '/testsession.(*Session).Enter',
             '/testsession.New'])]
dis_dir = OUT / 'disassembly'
dis_dir.mkdir(exist_ok=True)
for f in selected:
    result = subprocess.run(['/Library/Developer/CommandLineTools/usr/bin/llvm-objdump',
                             '-d', '--x86-asm-syntax=intel',
                             '--start-address=' + hex(f['address']),
                             '--stop-address=' + hex(f['end_address']), str(SOURCE)],
                            check=True, capture_output=True, text=True)
    lines = [f['name'], 'Virtual address: ' + hex(f['address']), '']
    for line in result.stdout.splitlines():
        call = re.search(r'\bcall\s+(0x[0-9a-f]+)', line)
        if call and int(call[1], 16) in by_address:
            line += ' ; ' + by_address[int(call[1], 16)]
        constant = re.search(r'# (0x[0-9a-f]+)', line)
        if constant:
            pos = offset(int(constant[1], 16), 8)
            if pos is not None:
                if 'sd' in line:
                    line += ' ; f64=' + repr(struct.unpack_from('<d', B, pos)[0])
                elif 'lea' in line:
                    preview = re.match(rb'[\x20-\x7e]{4,90}', B[pos:pos + 90])
                    if preview:
                        line += ' ; text_prefix=' + repr(preview[0].decode())
        lines.append(line)
    name = re.sub(r'[^A-Za-z0-9_.-]', '_', f['name'].split('/internal/')[-1])
    (dis_dir / (name + '.asm.txt')).write_text('\n'.join(lines) + '\n')

# Windows manifest is a PE resource; it is distinct from the Go build manifests.
rsrc = next((s for s in SECTIONS if s['name'] == '.rsrc'), None)
pe_resources = []
if rsrc:
    root_offset = rsrc['raw_offset']
    def resource_tree(relative, parts=()):
        directory = root_offset + relative
        nn, ni = struct.unpack_from('<HH', B, directory + 12)
        for j in range(nn + ni):
            key, value = struct.unpack_from('<II', B, directory + 16 + j * 8)
            if key & 0x80000000:
                p = root_offset + (key & 0x7fffffff)
                n = struct.unpack_from('<H', B, p)[0]
                key = B[p + 2:p + 2 + 2 * n].decode('utf-16le')
            path = (*parts, key)
            if value & 0x80000000:
                resource_tree(value & 0x7fffffff, path)
            else:
                data_rva, size, codepage, _ = struct.unpack_from('<4I', B, root_offset + value)
                pos = offset(BASE + data_rva, size)
                pe_resources.append(dict(path=list(path), size=size, file_offset=pos))
                if path[0] == 24 and pos is not None:
                    (OUT / 'windows_app.manifest').write_bytes(B[pos:pos + size])
    resource_tree(0)
save_json('pe_resources.json', pe_resources)
print(json.dumps(dict(size=len(B), compiler=compiler, embedded_resources=resources,
                     function_count=len(functions), app_function_count=len(app_functions),
                     app_modules=dict(sorted(module_counts.items())), json_literal_count=len(json_literals),
                     imports=[x['dll'] for x in imports]), ensure_ascii=False, indent=2))
