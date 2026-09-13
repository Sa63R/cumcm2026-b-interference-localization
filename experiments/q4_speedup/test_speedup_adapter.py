"""Offline checks through real SimulatorClient validation; never a network run."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from bootstrap import b,make_case,RoundedSimulator
from runtime import prepare_policy,run_policy
from test_adapter import SyntheticClient
import run_candidates as run_speedup
from run_candidates import prepare_speedup_policy,run_speedup_session
from simulator_client.errors import OutcomeUnknown
import online
import q4_coverage

ROOT=Path(__file__).resolve().parents[3]/'output/q4_speedup/schedule_dev/adapter_speedup'
CASES=[('all_omni',0.,'uniform','hash'),('all_directional',1.,'uniform','plus'),
       ('boundary_all_directional',1.,'outward_boundary','minus')]


class SpeedupAdapterTests(unittest.TestCase):
    def test_methods_full_sessions_with_true_emission_types(self):
        ROOT.mkdir(parents=True,exist_ok=True);results=[]
        with tempfile.TemporaryDirectory() as directory:
            for index,(label,fraction,placement,error) in enumerate(CASES):
                seed=236800000+index
                baseline_trace=None
                baseline_policy=None
                for method in ('baseline',)+run_speedup.METHODS:
                    if method=='combined' and index>0:continue
                    with self.subTest(method=method,case=label):
                        targets=make_case(seed,fraction,placement)
                        self.assertTrue(all(t.direction is None for t in targets) if fraction==0 else all(t.direction is not None for t in targets))
                        sim=RoundedSimulator(targets,seed,error,trace=True)
                        originals=(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_coverage.certify)
                        if method=='baseline':state,planner,geometry=prepare_policy('v4');metadata={}
                        else:state,metadata=prepare_speedup_policy(method,seed+71)
                        report=dict(metadata,status='not_entered',mode='synthetic_test_only')
                        with SyntheticClient(sim,Path(directory)/f'{method}_{index}.jsonl') as client:
                            if method=='baseline':
                                client.enter();report['policy']=run_policy(client,state,planner);client.exit()
                                baseline_trace=sim.trace[:]
                                baseline_policy=report['policy']
                            else:run_speedup_session(client,state,report)
                            self.assertEqual(client.state.session,'exited')
                            self.assertIsNone(client.pending_request)
                            self.assertEqual(client.state.cleared_count,len(targets))
                            self.assertEqual(sim.clear_count,len(targets))
                            self.assertAlmostEqual(client.state.virtual_time_s,sim.virtual_seconds)
                            if len(targets)<16:self.assertTrue(report['policy']['coverage_certificate']['ok'])
                            else:self.assertEqual(report['policy']['stop_certificate'],'source_upper_bound')
                            if method!='baseline':
                                self.assertEqual(report['status'],'policy_completed_and_exited')
                                self.assertEqual(report['method'],method)
                                self.assertEqual(report['version'],run_speedup.VERSION)
                                self.assertGreaterEqual(report['policy_cpu_seconds'],0)
                                self.assertGreater(report['policy_wall_seconds'],0)
                                self.assertGreater(report['compute_cache']['circle']['misses'],0)
                                self.assertTrue(report['coverage_cache_enabled'])
                                if len(targets)<16:self.assertGreater(report['coverage_compute_cache']['calls'],0)
                            if method=='prior_fixed':
                                self.assertEqual(state.planner.seed,seed+71)
                                self.assertGreater(report['policy']['planning']['attempts'],0)
                                self.assertEqual(report['policy']['planning'],state.planner.stats)
                            if method=='v4_fast':
                                self.assertEqual(sim.trace,baseline_trace)
                                normalize=lambda p:{k:({kk:vv for kk,vv in v.items() if kk!='wall'} if k=='coverage_certificate' and isinstance(v,dict) else v) for k,v in p.items()}
                                self.assertEqual(normalize(report['policy']),normalize(baseline_policy))
                        self.assertEqual(originals,(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_coverage.certify))
                        results.append(dict(method=method,scenario=label,seed=seed,success=True,
                                            purefast_trace_equal=sim.trace==baseline_trace if method=='v4_fast' else None,
                                            **sim.summary(),report=report))
        (ROOT/'results.json').write_text(json.dumps(dict(kind='OFFLINE SyntheticClient through real protocol validation; no official practice run',runs=results),indent=2))

    def test_missing_practice_assertion_never_constructs_client(self):
        with mock.patch.object(run_speedup,'SimulatorClient') as factory,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:run_speedup.main(['--robot-id','synthetic-only','--method','v4_fast'])
        self.assertEqual(error.exception.code,2);factory.assert_not_called()

    def test_main_writes_cpu_wall_version_and_cache_result(self):
        with tempfile.TemporaryDirectory() as directory:
            sim=RoundedSimulator(make_case(236890003,0.,'uniform'),236890003)
            def factory(robot_id,log_path):return SyntheticClient(sim,log_path)
            with mock.patch.object(run_speedup,'SimulatorClient',side_effect=factory),contextlib.redirect_stdout(io.StringIO()):
                folder=run_speedup.main(['--robot-id','synthetic-only','--method','v4_fast','--practice-confirmed',
                                         '--case-label','offline-main-check','--output',directory])
            report=json.loads((folder/'result.json').read_text())
            self.assertEqual(report['status'],'policy_completed_and_exited')
            self.assertEqual(report['version'],run_speedup.VERSION)
            self.assertEqual(report['method'],'v4_fast')
            self.assertTrue(report['cache_enabled'])
            self.assertTrue(report['coverage_cache_enabled'])
            self.assertIn('coverage_compute_cache',report)
            self.assertIn('coverage_compute_fast.py',report['source_sha256'])
            self.assertGreater(report['cpu_seconds'],0)
            self.assertGreater(report['wall_seconds'],0)
            self.assertIn('source_sha256',report)
            self.assertTrue((folder/'requests.jsonl').is_file())

    def test_uncertain_outcome_does_not_send_exit_or_new_action(self):
        class UnknownClient(SyntheticClient):
            def __init__(self,sim,path):
                super().__init__(sim,path);self.max_attempts=1;self.actions=[]
            def _exchange(self,action,timeout):
                self.actions.append((action.path,action.payload))
                if action.path=='/measure':raise OSError('simulated unresolved response')
                return super()._exchange(action,timeout)
        with tempfile.TemporaryDirectory() as directory:
            state,metadata=prepare_speedup_policy('v4_fast')
            sim=RoundedSimulator(make_case(236890001,1.,'uniform'),236890001)
            report=dict(metadata,status='not_entered')
            originals=(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_coverage.certify)
            with UnknownClient(sim,Path(directory)/'unknown.jsonl') as client:
                with self.assertRaises(OutcomeUnknown):run_speedup_session(client,state,report)
                self.assertEqual([p for p,_ in client.actions],['/enter','/measure'])
                self.assertEqual(report['pending_request'],client.pending_request)
                self.assertEqual(report['pending_request']['path'],'/measure')
                self.assertNotIn('error_exit_response',report)
                self.assertEqual(report['status'],'interrupted_or_failed')
                self.assertEqual(originals,(b.enclosing_circle,b.clip_halfplane,online.multi_route,q4_coverage.certify))

    def test_known_interruption_exits_active_session(self):
        with tempfile.TemporaryDirectory() as directory:
            state,metadata=prepare_speedup_policy('v4_fast')
            sim=RoundedSimulator(make_case(236890002,0.,'uniform'),236890002)
            report=dict(metadata,status='not_entered')
            def stop(state,client):raise RuntimeError('synthetic operator interruption')
            with SyntheticClient(sim,Path(directory)/'known.jsonl') as client:
                with self.assertRaises(RuntimeError):run_speedup_session(client,state,report,stop)
                self.assertEqual(client.state.session,'exited')
                self.assertIsNone(client.pending_request)
                self.assertIn('error_exit_response',report)


if __name__=='__main__':unittest.main(verbosity=2)
