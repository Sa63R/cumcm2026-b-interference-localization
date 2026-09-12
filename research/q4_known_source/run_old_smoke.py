"""Only the two previously opened 621 QA cases; preserve first raw and audit."""
from pathlib import Path
import gzip
import json
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from experiments import run_q4_per_source as runner
from experiments.run_q4_round2 import hashes, one
from experiments.audit_q4_known_source import audit_full, verify_source_contract
from experiments.q4_known_source_release import actual_service_count


def main():
    verify_source_contract()
    identity, inherited = runner.source_hashes(), hashes()
    spec_map = runner.read(ROOT/'research/q4_known_source/spec.json')
    runner.validate_spec(spec_map)
    label, spec = next(iter(spec_map.items()))
    output = ROOT/'research/q4_known_source/old-smoke'
    output.mkdir(exist_ok=False)
    manifest = dict(purpose='Previously opened cases for implementation QA only, no 635 selection',
        seeds=[621003, 621013], stage='pilot', label=label, spec=spec, source_sha256=identity)
    runner.write_new(output/'manifest.json', manifest)
    with zipfile.ZipFile(output/'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(identity): archive.writestr(path, (ROOT/path).read_bytes())
    rows = []
    for seed in manifest['seeds']:
        assert runner.source_hashes() == identity, 'Source changed before old QA case'
        record = one(seed, 'pilot', label, spec, inherited)
        raw = output/f'{label}-{seed}.json.gz'
        with gzip.open(raw, 'wt', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        try:
            audited = audit_full(record)
            count = actual_service_count(record)
            assert count == audited['prefix']['broad_service_actions']
            audited.update(input_sha256=runner.sha(raw))
        except Exception as error:
            audited = dict(passed=False, input_sha256=runner.sha(raw), error=str(error), traceback=traceback.format_exc())
            count = None
        runner.write_new(output/f'{seed}-first-audit.json', audited)
        r = record['row']
        row = dict(seed=seed, successful=r['successful'], source_total=r['source_total'],
            time_s=r['virtual_time_s'], time_per_source_s=r['virtual_time_s']/r['source_total'],
            lower_s=r['common_lower_bound_s'], T_LB=r['time_over_lower_bound'],
            audit_passed=audited['passed'], broad_service_actions=count)
        rows.append(row)
        print(json.dumps(row), flush=True)
    runner.write_new(output/'summary.json', dict(rows=rows, source_unchanged=runner.source_hashes()==identity))
    return int(not all(r['successful'] and r['audit_passed'] for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
