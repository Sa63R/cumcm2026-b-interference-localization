"""Verify and export a practice observation record, omitting account identifiers."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from experiments.audit_q4_joint_visibility_prefix import audit_joint_visibility_prefix
from experiments.audit_q4_clear_before_probe import audit_clear_before_probe_prefix

SPEC = {'entrypoint': 'strategies.q4_joint_visibility:run_q4_joint_visibility',
        'kwargs': {'config': 'probe', 'max_expansions': 200}}


def verify(directory, output):
    envelope = json.loads((directory/'summary.json').read_bytes())
    bounds = json.loads((directory/'lower_bounds.json').read_bytes())
    if (envelope['problem'] != 4 or envelope['declared_mode'] != 'practice'
            or envelope['completed'] is not True or envelope['method_metadata']['spec'] != SPEC
            or bounds['official_all_clear_verified'] is not True):
        raise ValueError('Need a completed and registered qualified Q4 practice')
    search = envelope['search']
    requests, accepted, history = {}, set(), []
    allowed = ('accepted', 'virtual_time_s', 'measure_result', 'svd_deg', 'clear_result',
               'max_virtual_duration_s', 'max_real_duration_s')
    for line in (directory/'requests.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        if item['event'] == 'request':
            requests[item['payload']['request_id']] = item
        elif item['event'] == 'response':
            request = requests[item['request_id']]
            if request['attempt'] != item['attempt']:
                raise ValueError('Request/response attempt mismatch')
            if item['response'].get('accepted') is True:
                if item['request_id'] in accepted:
                    raise ValueError('Duplicate accepted response needs explicit replay handling')
                accepted.add(item['request_id'])
            response = {k: item['response'][k] for k in allowed if k in item['response']}
            entry = dict(action=request['path'], response=response)
            for key in ('channel', 'position'):
                if key in request['payload']:
                    entry[key] = request['payload'][key]
            history.append(entry)
    record = dict(summary=search, history=history, spec=SPEC, row={'successful': True},
        origin='Official Q4 practice; account/robot/arena/request identifiers removed')
    inherited = audit_clear_before_probe_prefix(record)
    joint = audit_joint_visibility_prefix(record)
    if search['cleared_count'] != bounds['official_source_total']:
        raise ValueError('Registered source total and actual clears differ')
    if abs(search['virtual_time_s'] - bounds['actual_virtual_time_s']) > 1e-6:
        raise ValueError('Bound applies to a different elapsed task time')
    result = dict(problem=4, mode='practice', source_total=bounds['official_source_total'],
        cleared_total=search['cleared_count'], virtual_time_s=search['virtual_time_s'],
        historical_lower_bound_s=bounds['conditional_guaranteed_all_clear_lower_s'],
        time_over_historical_lower_bound=bounds['time_to_conditional_lower_bound_ratio'],
        official_all_clear_verified=True, inherited_r8_audit=inherited, joint_prefix_audit=joint,
        raw_input_sha256={name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                          for name in ('summary.json','requests.jsonl','registration.json','lower_bounds.json')},
        limitations='One official practice. No same-case RL run; paired RL performance is reported separately on the independent local scenes. Bound uses the historical conditional containing-disk relaxation.')
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(output/'observed-record.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('source_total','cleared_total','virtual_time_s',
        'historical_lower_bound_s','time_over_historical_lower_bound','official_all_clear_verified')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.input, args.output)
