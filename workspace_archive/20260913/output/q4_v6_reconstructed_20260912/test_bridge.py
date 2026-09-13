"""Adapter semantics and equality against the published V4 execution kernel."""
import json
from pathlib import Path
import sys
import unittest
import benchmark as bench
from jammers_local.core import Scenario, Session, Source, LocalClient


def normalized(session):
    return [(r['path'],r['request'],{k:v for k,v in r['response'].items()
            if k not in ('real_timestamp_ms','remaining_real_duration_s')}) for r in session.history]


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bench.initialize(1.005)

    def test_deferred_move_clear_channel_and_costs(self):
        scene=Scenario(4,(Source(1,0,0,1_000_000_000,'omni',None),
                         Source(2,100_000_000,0,1_000_000_000,'directional',0)),77,profile='fixture')
        session=Session(scene);client=LocalClient(session.dispatch);client.enter();device=bench.Device(client)
        device.move((100.,0.))
        self.assertEqual(session.engine.position,(0.,0.))
        self.assertTrue(device.clear(2))
        self.assertEqual(device.channel,1)
        self.assertEqual(session.engine.channel,1)
        self.assertEqual(client.virtual_time_s,25.)
        self.assertEqual(device.detect(2).status,'none')
        self.assertEqual(device.channel,2)
        self.assertEqual(client.virtual_time_s,31.)
        device.move((0.,0.))
        self.assertEqual(device.detect(1).status,'strong')
        self.assertEqual(client.virtual_time_s,57.)
        self.assertTrue(device.clear(1))
        self.assertEqual(client.virtual_time_s,62.)

    def test_published_v4_full_trace_equivalence(self):
        published=bench.ROOT.parents[1]/'tmp/q4-v4-publish/experiments/q4_comparison'
        sys.path.insert(0,str(published))
        from online import State
        plan=json.loads((bench.ROOT/'plan.json').read_text())
        evidence=[]
        for case in [plan['cases'][0],plan['cases'][300],plan['cases'][335]]:
            with self.subTest(case=case['key']):
                scene=Scenario.from_dict(case['scenario'])
                sessions=[]
                for method in ('vendor','published'):
                    session=Session(scene);client=LocalClient(session.dispatch);client.enter();device=bench.Device(client)
                    if method=='vendor':
                        report=bench.solve_v4(device)
                    else:
                        state=State(analytic=False)
                        while state.prepare(): state.execute(device,state.ordered_actions(device)[0])
                        report=state.report()
                    client.exit();bench.audit(session,report);sessions.append(session)
                self.assertEqual(normalized(sessions[0]),normalized(sessions[1]))
                evidence.append(dict(case_key=case['key'],actions=len(sessions[0].history),
                    virtual_time_us=sessions[0].engine.time_us,trace_equal=True))
        bench.dump(bench.ROOT/'published_v4_equivalence.json',evidence)


if __name__=='__main__': unittest.main(verbosity=2)
