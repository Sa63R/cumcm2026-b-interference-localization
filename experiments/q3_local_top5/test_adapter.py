"""Cross-check the fast local adapter against the existing official-protocol client."""
from pathlib import Path
import sys
import tempfile
import unittest
import runner as r

sys.path.insert(0,str(r.HERE.parent/'q3_official_practice'))
from runtime import SimulatorClient,run_policy

class ProtocolClient(SimulatorClient):
    def __init__(self,dispatch,path):
        self.__dispatch=dispatch
        super().__init__('local-test',base_url='http://127.0.0.1:1',log_path=path)
    def _exchange(self,action,timeout):
        return self.__dispatch(action.path,action.payload)

def actions(data):
    return [(a['path'],a['request'].get('position'),a['request'].get('channel'),
             {k:v for k,v in a['response'].items() if k not in ('real_timestamp_ms','remaining_real_duration_s')})
            for a in data['history']]

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.initialize()
    def test_all_five_match_existing_protocol_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(2):
                scene=r.Scenario.generate(3,f'q3-top5-adapter-{seed}')
                for method in r.METHODS:
                    with self.subTest(seed=seed,method=method):
                        fast,fast_data=r.run_one(scene,method)
                        session=r.Session(scene)
                        with ProtocolClient(session.dispatch,Path(tmp)/f'{seed}_{method}.jsonl') as client:
                            client.enter();result=run_policy(client,method);client.exit()
                            self.assertTrue(result['complete_channel_certificate'])
                            self.assertAlmostEqual(client.state.time_breakdown.total_s,session.engine.time_us/1e6,places=4)
                        self.assertTrue(fast['success'],fast['error'])
                        self.assertEqual(actions(fast_data),actions(session.export()))
    def test_move_cost_clear_failure_and_radio_state(self):
        from jammers_local.core import Source
        scene=r.Scenario(3,(Source(1,300_000_000,400_000_000,1_000_000_000,'omni',None),),1,profile='fixture')
        session=r.Session(scene);client=r.LocalClient(session.dispatch);device=r.Device(client)
        client.enter();device.move((300,400))
        self.assertEqual(client.position,(0.,0.))
        self.assertEqual(device.detect(3).kind,'none');self.assertEqual(client.virtual_time_s,106.)
        self.assertTrue(device.clear(1));self.assertFalse(device.clear(1))
        self.assertEqual(device.channel,3);self.assertEqual(client.virtual_time_s,114.)
        client.exit();self.assertTrue(r.replay(session.export())['matched'])

if __name__=='__main__':unittest.main(verbosity=2)
