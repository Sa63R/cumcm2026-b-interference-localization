"""Read selected actual probes; preserve the existing descriptive diagnostic."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]


def main():
    result={'scope':'Descriptive predicted no-signal probabilities on the selected actual replacement probes; clustered, selected samples and assumed prior, not a calibrated official probability or a tuning set.','splits':{}}
    for split in ('development','development-stress'):
        rows=[]
        for path in (ROOT/'results/q4_anchor_first_hit'/split/'records').glob('*.json.gz'):
            record=json.loads(gzip.decompress(path.read_bytes()))
            for event in record['summary']['strategy_parameters']['anchor_first_hit_log']:
                if not event['executed_measure']:continue
                model=event['model'];score=model['candidate_scores'][model['selected_index']]
                p=sum(b['mass'] for b in score['branches'] if b['outcome']==['no_signal'])
                rows.append(dict(seed=record['row']['seed'],event_id=event['id'],predicted_no_signal_probability=p,
                    actual_no_signal=event['actual_result']=='no_signal',raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        result['splits'][split]=dict(probes=len(rows),predicted_mean_no_signal=statistics.mean(r['predicted_no_signal_probability'] for r in rows),
            actual_no_signal_fraction=statistics.mean(r['actual_no_signal'] for r in rows),details=rows)
    output=ROOT/'research/q4_anchor_first_hit/feedback-calibration.json'
    if output.exists():
        saved=json.loads(output.read_bytes())
        assert saved['scope']==result['scope']
        for split in result['splits']:
            a,b=saved['splits'][split],result['splits'][split]
            assert {k:v for k,v in a.items() if k!='details'}=={k:v for k,v in b.items() if k!='details'}
            assert sorted(a['details'],key=lambda r:(r['seed'],r['event_id']))==sorted(b['details'],key=lambda r:(r['seed'],r['event_id']))
        print('Existing descriptive diagnostic exactly reproduced; original file preserved.')
    else:
        with output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
