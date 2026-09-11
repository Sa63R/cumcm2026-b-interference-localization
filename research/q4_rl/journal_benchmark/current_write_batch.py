from pathlib import Path
import gzip
import json

def _write_batch(path, value):
    temporary = Path(str(path) + ".tmp")
    # dumps uses the native JSON encoder; dump emits millions of tiny writes.
    # The decoded bytes are identical, including float spelling and separators.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with gzip.open(temporary, "wb", compresslevel=6) as stream:
        stream.write(payload)
    temporary.replace(path)
