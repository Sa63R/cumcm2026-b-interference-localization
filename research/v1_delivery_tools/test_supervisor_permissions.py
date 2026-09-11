"""Linux may hide cwd for non-dumpable processes owned by the same user."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def process_reader(tmp_path, monkeypatch):
    path = Path(__file__).with_name('run_serial_benchmarks.py')
    spec = importlib.util.spec_from_file_location('serial_permissions_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proc = tmp_path / '123'
    proc.mkdir()
    (proc / 'stat').write_text('123 (python) S 0')
    resolve = Path.resolve

    def denied_cwd(self, *args, **kwargs):
        if self == proc / 'cwd':
            raise PermissionError('synthetic non-dumpable process cwd')
        return resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', denied_cwd)
    return module, proc, tmp_path


def test_known_non_training_process_with_hidden_cwd_is_not_training(process_reader):
    module, proc, root = process_reader
    (proc / 'cmdline').write_bytes(b'python\0-m\0unrelated.service\0')
    result = module.read_process(123, root)
    assert result['state'] == 'S' and result['cwd'] is None
    assert not module.is_training_command(result['command'])


def test_live_training_with_hidden_cwd_still_refuses_to_proceed(process_reader):
    module, proc, root = process_reader
    (proc / 'cmdline').write_bytes(b'python\0-m\0research_rl.train\0')
    with pytest.raises(PermissionError):
        module.read_process(123, root)


def test_zombie_training_needs_no_working_directory(process_reader):
    module, proc, root = process_reader
    (proc / 'cmdline').write_bytes(b'python\0-m\0research_rl.train\0')
    (proc / 'stat').write_text('123 (python) Z 0')
    assert module.read_process(123, root)['state'] == 'Z'
