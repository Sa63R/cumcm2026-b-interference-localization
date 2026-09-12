"""Read frozen JSON sessions directly from their lossless ZIP archive."""
import json
from pathlib import Path
import zipfile


def iter_sessions(folder):
    folder = Path(folder)
    archive = folder/'sessions.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            for name in sorted(bundle.namelist()):
                if name.endswith('.json'):
                    yield name, json.loads(bundle.read(name))
    else:
        for path in sorted((folder/'sessions').glob('*.json')):
            yield path.name, json.loads(path.read_text())


def read_session(folder, name):
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('Expected a session JSON basename')
    folder = Path(folder)
    archive = folder/'sessions.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as bundle:
            return json.loads(bundle.read(name))
    return json.loads((folder/'sessions'/name).read_text())
