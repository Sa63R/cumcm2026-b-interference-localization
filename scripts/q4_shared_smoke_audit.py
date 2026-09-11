"""Posthoc public-prefix cost audit; no alternate-action feedback is simulated."""
from itertools import combinations
import gzip
import hashlib
import json
from pathlib import Path
import sys
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from q4_rl.shared_micro_controller import Q4SharedMicroSearch
from q4_rl.shared_cover import position_key,public_plan_cost,replay_shared_cover
from simulator_client.state import ClientState,Position
from experiments.q4_rl_evaluate import report_rows


class StrictObservedPrefixClient:
    """Only replays the exact already recorded action at its original position."""
    def __init__(self,history):
        self.state=ClientState()
        self.pending_request=None
        self.remaining_real_time_s=None
        self.history=history
        self.index=0

    def reply(self,kind,position,channel):
        item=self.history[self.index]
        assert (kind,position_key(position),channel)==(item['action'],position_key(item['position']),item['channel'])
        self.index+=1
        self.state.position=position
        self.state.virtual_time_s=item['virtual_time_s']
        if kind=='measure': self.state.current_channel=channel
        result=dict(accepted=True)
        result['measure_result' if kind=='measure' else 'clear_result']=item['result']
        if item['result']=='direction': result['svd_deg']=item['bearing_deg']
        return result

    def measure(self,p,c): return self.reply('measure',p,c)
    def clear(self,p,c): return self.reply('clear',p,c)


def main():
    folder=ROOT/'results/q4_rl/shared-micro-smoke'
    paths=sorted((folder/'records').glob('*.gz'))
    records=[json.loads(gzip.decompress(p.read_bytes())) for p in paths]
    keyed={(r['row']['case_id'],r['row']['strategy']):r for r in records}
    diagnostics=[]
    for record in records:
        if record['row']['strategy']!='shared512': continue
        metrics=record['summary']['learning']
        history=record['summary']['action_history']
        micro=keyed[(record['row']['case_id'],'micro512')]
        semantic=[]
        for source in (record['history'],micro['history']):
            normalized=deepcopy(source)
            for action in normalized:
                action['response'].pop('real_timestamp_ms',None)
            semantic.append(normalized)
        exact_replay=replay_shared_cover(history,metrics['shared_artifact'])
        assert exact_replay['complete']
        reviews=[]
        for review in metrics['shared_plan_reviews']:
            prefix=history[:review['after_action_index']+1]
            client=StrictObservedPrefixClient(prefix)
            search=Q4SharedMicroSearch(client,max_shared_reviews=0,record_transitions=False)
            for item in prefix:
                search._perform(item['action'],item['position'],item['channel'],item['phase'])
            assert search.report.action_history==prefix
            base=super(Q4SharedMicroSearch,search)._candidates()
            points=search._public_service_candidates(base)
            actions=search._public_schedule(points,base)
            current=client.state.position
            before=public_plan_cost(actions,current,client.state.current_channel)
            assert before==review['full_proxy_cost']
            search.shared._proxy_channel=client.state.current_channel
            visited={position_key(a['position']) for a in prefix}|{position_key(current)}
            candidates=[]
            for station in search.points:
                key=position_key(station)
                if abs(station.distance_to(Position(0,0))-970.)>1e-6 or key in visited: continue
                owed=tuple(sorted(c for c in search._unknown() if key in search.shared.pending[c]))
                if not owed: continue
                for count in (1,2):
                    for subset in combinations(points,count):
                        replacement=search.shared._replacement_actions(station,subset,owed,actions,current)
                        if replacement is None: continue
                        after=public_plan_cost(replacement,current,client.state.current_channel)
                        saving=before['total_upper_s']-after['total_upper_s']
                        candidates.append(dict(station=list(key),point_count=count,
                            points=[list(position_key(p.position)) for p in subset],
                            saved_proxy_s=saving,cost_positive=saving>1e-6))
            reviews.append(dict(after_action_index=review['after_action_index'],candidate_costs=candidates,
                one_point_cost_positive=sum(c['point_count']==1 and c['cost_positive'] for c in candidates),
                two_point_cost_positive=sum(c['point_count']==2 and c['cost_positive'] for c in candidates)))
        diagnostics.append(dict(case_id=record['row']['case_id'],
            actual_micro_history_identical=history==micro['summary']['action_history'],
            actual_wire_history_identical=record['history']==micro['history'],
            wire_history_equal_excluding_real_timestamp=semantic[0]==semantic[1],
            shared_packet_scans=metrics['shared_service_actual_measurements'],
            certificate_cost=metrics['shared_artifact']['cost'],
            exact_final_replay_wall_s=metrics['shared_completion_replay_wall_s'],
            cancelled_stations=metrics['shared_artifact']['cancelled_station_count'],
            actual_final_replay=exact_replay,public_prefix_cost_reviews=reviews))
    comparison=report_rows([r['row'] for r in records],reference='micro512',samples=2000)
    output=dict(method='Strict exact observed prefixes, public schedule arithmetic only; no counterfactual feedback or policy evaluation',
        record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostics=diagnostics,paired_against_micro=comparison['paired'])
    target=folder/'shared_posthoc_audit.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=[dict(case_id=d['case_id'],same_actual_history=d['actual_micro_history_identical'],
        same_wire_history_excluding_real_timestamp=d['wire_history_equal_excluding_real_timestamp'],one_positive=[r['one_point_cost_positive'] for r in d['public_prefix_cost_reviews']],
        two_positive=[r['two_point_cost_positive'] for r in d['public_prefix_cost_reviews']]) for d in diagnostics],
        paired=comparison['paired']['shared512']),ensure_ascii=False))


if __name__=='__main__': main()
