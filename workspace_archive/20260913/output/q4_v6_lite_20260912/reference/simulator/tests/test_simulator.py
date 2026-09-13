import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import http.client
import json
import math
from pathlib import Path
import random
import threading
import unittest

from jammers_local.algorithms import CounterSource, round_away, error_degrees, quantize_bearing, move_microseconds
from jammers_local.core import Scenario,Source,Session,Engine,LocalClient
from jammers_local.server import Application,Server,decode_json
from jammers_local.__main__ import replay


def fixture(directional=False,x=0,y=0,radius=1000):
    return Scenario(4 if directional else 3,
                    (Source(1,round(x*1e6),round(y*1e6),round(radius*1e6),
                            'directional' if directional else 'omni',0 if directional else None),),
                    20260912,profile='fixture')


def payload(rid,position=None,channel=1):
    p=dict(arena_id='default',robot_id='local-test',request_id=rid)
    if position is not None: p.update(position=dict(x=position[0],y=position[1]),channel=channel)
    return p


class RuleTests(unittest.TestCase):
    def test_counter_hmac_vector_and_label_isolation(self):
        seed=bytes(range(32))
        msg=b'practice-case-v1\0count\0'+bytes(8)
        expected=int.from_bytes(hmac.new(seed,msg,hashlib.sha256).digest()[:8],'big')
        a,b=CounterSource(seed),CounterSource(seed)
        self.assertEqual(a.next('count'),expected)
        a.next('radius')
        self.assertEqual(a.next('channels'),b.next('channels'))

    def test_generator_rules_and_all_directional_possible(self):
        found_all=False
        for i in range(200):
            q3,q4=Scenario.generate(3,str(i)),Scenario.generate(4,str(i))
            self.assertEqual(q4,Scenario.generate(4,str(i)))
            self.assertEqual(q3.noise_seed,q4.noise_seed)
            self.assertTrue(10<=len(q4.sources)<=16)
            found_all |= all(s.kind=='directional' for s in q4.sources)
            for a,b in zip(q3.sources,q4.sources):
                self.assertEqual((a.x_um,a.y_um,a.channel,a.max_receive_um),(b.x_um,b.y_um,b.channel,b.max_receive_um))
                self.assertLessEqual(b.x_um*b.x_um+b.y_um*b.y_um,1_770_000_000**2)
        self.assertTrue(found_all)

    def test_scenario_roundtrip_and_reject_invalid(self):
        s=Scenario.generate(4,'roundtrip')
        self.assertEqual(Scenario.from_dict(s.as_dict()),s)
        d=s.as_dict();d['sources'][0]['x_um']=1_800_000_000
        with self.assertRaises(ValueError): Scenario.from_dict(d)
        with self.assertRaises(ValueError): Scenario(True,s.sources,s.noise_seed)

    def test_rounding_and_movement(self):
        self.assertEqual([round_away(x) for x in [-1.5,-.5,.5,1.5] ],[-2,-1,1,2])
        self.assertEqual(round_away(math.nextafter(.5,0)),0)
        self.assertEqual(move_microseconds(0,0,3,4),1_000_000)
        self.assertEqual(move_microseconds(0,0,0,0),0)

    def test_noise_bounds_continuity_and_quantization(self):
        rng=random.Random(9)
        for _ in range(1000):
            x,y=rng.uniform(-2000,2000),rng.uniform(-2000,2000)
            e=error_degrees(77,3,x,y)
            self.assertEqual(e,error_degrees(77,3,x,y))
            self.assertLessEqual(abs(e),1+1e-14)
            t=rng.uniform(0,360)
            q=quantize_bearing(t,e)
            self.assertTrue(0<=q<360)
            self.assertLessEqual(abs(math.remainder(q-t,360)),1+1e-12)
        for x in [-300,-150,0,150,300]:
            self.assertLess(abs(error_degrees(77,3,x-1e-6,73)-error_degrees(77,3,x+1e-6,73)),1e-6)
        self.assertEqual(quantize_bearing(12.346,.9999),13.34)
        self.assertEqual(quantize_bearing(359.8,.5),.3)


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.session=Session(fixture())
        self.session.dispatch('/enter',payload('enter'))

    def call(self,path,rid,pos,c=1):
        status,response=self.session.dispatch(path,payload(rid,pos,c))
        self.assertEqual(status,200)
        self.assertTrue(response['accepted'])
        return response

    def test_attachment_timing_example(self):
        self.assertEqual(self.call('/measure','m1',(300,400))['virtual_time_s'],105)
        self.assertEqual(self.call('/measure','m2',(300,400),2)['virtual_time_s'],111)
        self.assertEqual(self.call('/clear','c',(300,0),3)['virtual_time_s'],194)
        self.assertEqual(self.call('/measure','m3',(300,0),2)['virtual_time_s'],199)
        self.assertEqual(self.session.engine.position,(300.,0.))

    def test_backside_near_and_clear(self):
        self.session=Session(fixture(True))
        self.session.dispatch('/enter',payload('enter'))
        self.assertEqual(self.call('/measure','back',(-3,0))['measure_result'],'no_signal')
        self.assertEqual(self.call('/measure','front',(3,0))['measure_result'],'near')
        self.assertEqual(self.call('/clear','clear',(-20,0))['clear_result'],'success')
        self.assertEqual(self.call('/measure','gone',(3,0))['measure_result'],'no_signal')
        self.assertEqual(self.call('/clear','again',(0,0))['clear_result'],'no_target_in_range')

    def test_directional_zero_distance_and_edge(self):
        s=fixture(True).sources[0]
        self.assertTrue(Engine.covered(s,0,0,0))
        self.assertTrue(Engine.covered(s,0,500,500))
        self.assertFalse(Engine.covered(s,-.001,500,500))

    def test_receive_near_and_clear_inclusive_boundaries(self):
        self.assertEqual(self.call('/measure','edge',(1000,0))['measure_result'],'direction')
        self.assertEqual(self.call('/measure','outside',(1000.000001,0))['measure_result'],'no_signal')
        self.assertEqual(self.call('/measure','near',(5,0))['measure_result'],'near')
        self.assertEqual(self.call('/measure','notnear',(5.000001,0))['measure_result'],'direction')
        self.assertEqual(self.call('/clear','fail',(20.000001,0))['clear_result'],'no_target_in_range')
        self.assertEqual(self.call('/clear','success',(20,0))['clear_result'],'success')

    def test_repeat_readings_and_clear_not_switching(self):
        a=self.call('/measure','a',(500,400))
        b=self.call('/measure','b',(500,400))
        self.assertEqual(a['svd_deg'],b['svd_deg'])
        self.call('/clear','c',(500,400),8)
        self.assertEqual(self.session.engine.channel,1)
        self.assertEqual(self.session.engine.switches,0)

    def test_no_signal_updates_channel_and_invalid_action_is_free(self):
        self.call('/measure','missing',(0,0),20)
        before=self.session.engine.snapshot()
        status,r=self.session.dispatch('/measure',payload('bad',(math.inf,0),1))
        self.assertEqual(status,400);self.assertFalse(r['accepted'])
        self.assertEqual(self.session.engine.snapshot(),before)
        self.assertEqual(self.session.engine.channel,20)

    def test_idempotency_semantic_numeric_identity_and_conflict(self):
        req=payload('same',(-0.,10),1)
        a=self.session.dispatch('/measure',req)
        req['position']['x']=0;req['channel']=1.0
        self.assertEqual(self.session.dispatch('/measure',req),a)
        self.assertEqual(self.session.engine.measurements,1)
        req['position']['x']=1
        self.assertEqual(self.session.dispatch('/measure',req)[0],409)
        self.assertEqual(self.session.dispatch('/clear',payload('same',(0,10)))[0],409)

    def test_unknown_fields_dont_reserve_id(self):
        p=payload('fix',(0,0));p['typo']=1
        self.assertEqual(self.session.dispatch('/measure',p)[1]['accepted'],False)
        del p['typo']
        self.assertTrue(self.session.dispatch('/measure',p)[1]['accepted'])

    def test_exit_replay_and_ended_reject(self):
        req=payload('exit')
        first=self.session.dispatch('/exit',req)
        self.assertEqual(self.session.dispatch('/exit',req),first)
        self.assertFalse(self.session.dispatch('/enter',payload('new'))[1]['accepted'])

    def test_exact_virtual_timeout_after_action(self):
        session=Session(fixture(),max_virtual_us=5_000_000)
        session.dispatch('/enter',payload('e'))
        self.assertTrue(session.dispatch('/measure',payload('m',(0,0)))[1]['accepted'])
        self.assertEqual(session.engine.stop_reason,'virtual_timeout')
        self.assertFalse(session.dispatch('/exit',payload('x'))[1]['accepted'])

    def test_real_window_and_program_deadlines(self):
        now=[0.]
        s=Session(fixture(),clock=lambda:now[0],window_s=1500,max_real_s=1200)
        now[0]=400
        self.assertEqual(s.dispatch('/enter',payload('e'))[1]['remaining_real_duration_s'],1100)
        now[0]=1500
        self.assertFalse(s.dispatch('/measure',payload('late',(0,0)))[1]['accepted'])
        self.assertEqual(s.engine.stop_reason,'window_timeout')
        now[0]=0
        s=Session(fixture(),clock=lambda:now[0])
        s.dispatch('/enter',payload('e'))
        now[0]=1200
        self.assertFalse(s.dispatch('/exit',payload('late'))[1]['accepted'])
        self.assertEqual(s.engine.stop_reason,'program_timeout')

    def test_concurrent_same_waits_different_conflicts(self):
        entered,release=threading.Event(),threading.Event()
        original=self.session.engine.apply
        def delayed(path,p):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(path,p)
        self.session.engine.apply=delayed
        p=payload('one',(0,0))
        with ThreadPoolExecutor(3) as pool:
            first=pool.submit(self.session.dispatch,'/measure',p)
            self.assertTrue(entered.wait(1))
            duplicate=pool.submit(self.session.dispatch,'/measure',p)
            different=self.session.dispatch('/measure',payload('two',(1,1)))
            self.assertEqual(different[0],409)
            release.set()
            self.assertEqual(first.result(),duplicate.result())
        self.assertEqual(self.session.engine.measurements,1)

    def test_replay_export_and_detect_tampering(self):
        self.call('/measure','m',(123.45,345.67))
        self.call('/clear','c',(20,0))
        self.session.dispatch('/exit',payload('x'))
        data=self.session.export()
        self.assertTrue(replay(data)['matched'])
        data['history'][1]['response']['virtual_time_s']+=1
        self.assertFalse(replay(data)['matched'])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.app=Application(fixture())
        self.server=Server(self.app,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,path,body=None,method='POST',headers=None):
        c=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
        raw=json.dumps(body).encode() if isinstance(body,dict) else body
        c.request(method,path,raw,headers=headers or {'Content-Type':'application/json'})
        r=c.getresponse();status=r.status;data=r.read();c.close()
        return status,json.loads(data)

    def test_four_endpoint_roundtrip_has_no_truth_fields(self):
        self.assertTrue(self.request('/enter',payload('e'))[1]['accepted'])
        r=self.request('/measure',payload('m',(100,100)))[1]
        self.assertEqual(set(r),{'accepted','real_timestamp_ms','virtual_time_s','measure_result','svd_deg'})
        self.assertEqual(self.request('/clear',payload('c',(0,0)))[1]['clear_result'],'success')
        self.assertEqual(self.request('/exit',payload('x'))[1]['exit_reason'],'user_exit')
        self.assertNotIn('scenario',self.request('/local/state',method='GET')[1])
        self.assertIn('scenario',self.request('/local/state?truth=1',method='GET')[1])

    def test_transport_rejections(self):
        base=json.dumps(payload('e')).encode()
        for path,body,status in [('/enter/',base,404),('/enter?x=1',base,404),
                                ('/enter',b'{"a":1,"a":2}',400),('/enter',b'\xef\xbb\xbf'+base,400),
                                ('/enter',b'NaN',400),('/enter',b'[]',400),('/enter',b'x'*65537,413)]:
            self.assertEqual(self.request(path,body)[0],status)
        self.assertEqual(self.request('/enter',method='GET')[0],405)
        self.assertEqual(self.request('/enter',base,headers={'Content-Type':'text/plain'})[0],415)
        self.assertEqual(self.request('/enter',base,headers={'Content-Type':'application/json; x=y'})[0],415)
        self.assertEqual(self.request('/enter',base,headers={'Content-Type':'application/json','Content-Encoding':'gzip'})[0],415)

    def test_validation_rejections_and_identifier(self):
        for mutate in [lambda p:p.update(robot_id='bad\u200b'),lambda p:p.update(request_id=''),
                       lambda p:p.update(channel=True),lambda p:p.update(channel=1.5),
                       lambda p:p.update(position={'x':True,'y':0}),lambda p:p.update(position={'x':2000001,'y':0})]:
            p=payload('a',(0,0));mutate(p)
            self.assertEqual(self.request('/measure',p)[0],400)
        wrong=payload('same');wrong['robot_id']='someone-else'
        self.assertFalse(self.request('/enter',wrong)[1]['accepted'])
        self.assertTrue(self.request('/enter',payload('same'))[1]['accepted'])

    def test_depth(self):
        value={}
        for _ in range(16): value={'a':value}
        self.assertEqual(self.request('/enter',value)[0],400)


if __name__=='__main__': unittest.main()
