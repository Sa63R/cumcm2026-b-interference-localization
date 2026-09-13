"""Loopback HTTP protocol and a separate local inspection interface."""
from __future__ import annotations
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from .core import Session, Scenario, LocalClient, PATHS, MAX_BODY
from .baseline import run as run_baseline


def decode_json(raw):
    if raw.startswith(b'\xef\xbb\xbf'): raise ValueError('BOM is not allowed')
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result: raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def reject(value): raise ValueError('non-finite JSON literal')
    data=json.loads(raw.decode('utf-8'),object_pairs_hook=unique,parse_constant=reject)
    def check(value,depth=1):
        if isinstance(value,(dict,list)):
            if depth>16: raise ValueError('JSON depth exceeds 16')
            for child in (value.values() if isinstance(value,dict) else value): check(child,depth+1)
    check(data)
    if not isinstance(data,dict): raise ValueError('JSON object required')
    return data


class Application:
    def __init__(self,scenario,robot_id='local-test',*,results_dir=None,**session_options):
        self.robot_id=robot_id
        self.session_options=session_options
        self.session=Session(scenario,robot_id=robot_id,**session_options)
        self.results_dir=Path(results_dir) if results_dir else None
        self.lock=threading.RLock()
        self.demo_thread=None
        self.demo_error=None
        self.cancel=threading.Event()
        self.last_export=None

    def running(self): return self.demo_thread is not None and self.demo_thread.is_alive()

    def save(self):
        result=self.session.export()
        if self.results_dir:
            self.results_dir.mkdir(parents=True,exist_ok=True)
            filename=f'run-{time.time_ns()}-{self.session.engine.scenario.case_id}.json'
            target=self.results_dir/filename
            target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            self.last_export=str(target.resolve())
        return result

    def reset(self,data):
        with self.lock:
            if self.running(): raise RuntimeError('请先停止演示，再新建场景。')
            scenario=Scenario.from_dict(data['scenario']) if 'scenario' in data else Scenario.generate(
                data.get('problem',4),data.get('seed','demo-20260912'),seed_hex=data.get('seed_hex'))
            self.save()
            self.session=Session(scenario,robot_id=self.robot_id,**self.session_options)
            self.cancel=threading.Event()
            self.demo_error=None

    def demo(self):
        with self.lock:
            if self.running(): raise RuntimeError('演示正在运行。')
            if self.session.snapshot()['lifecycle']!='ready': raise RuntimeError('演示需要一个尚未进入的新场景。')
            session=self.session
            self.cancel=threading.Event()
            self.demo_error=None
            def exchange(path,payload):
                result=session.dispatch(path,payload)
                self.cancel.wait(.006)  # Visual pacing only; no virtual-time change.
                return result
            client=LocalClient(exchange,self.robot_id,cancel=self.cancel)
            def work():
                try: run_baseline(client,session.engine.scenario.problem_no)
                except Exception as exc:
                    self.demo_error=str(exc)
                    if session.engine.lifecycle!='ended': session.engine.finish('demo_stopped')
                finally:
                    try: self.save()
                    except OSError as exc: self.demo_error=f'导出失败：{exc}'
            self.demo_thread=threading.Thread(target=work,daemon=True,name='local-jammers-demo')
            self.demo_thread.start()

    def state(self,truth=False):
        with self.lock:
            state=self.session.snapshot(truth)
            state['seed_label']=self.session.engine.scenario.label
            with self.session._lock:
                state['history']=deepcopy(self.session.history[-4000:])
            state.update(demo_running=self.running(),demo_error=self.demo_error,last_export=self.last_export)
            return state


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,app,port=2027):
        self.app=app
        super().__init__(('127.0.0.1',port),Handler)


class Handler(BaseHTTPRequestHandler):
    server_version='LocalJammers/0.1'
    def log_message(self,*args): pass

    @property
    def app(self): return self.server.app

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def send(self,status,value,ctype='application/json; charset=utf-8',download=None):
        raw=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Cache-Control','no-store')
        if download: self.send_header('Content-Disposition',f'attachment; filename="{download}"')
        self.end_headers()
        if self.command!='HEAD':
            try: self.wfile.write(raw)
            except (BrokenPipeError,ConnectionResetError): pass

    def local_origin(self):
        port=self.server.server_port
        hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host') not in hosts: return False
        origin=self.headers.get('Origin')
        return origin is None or origin in {f'http://{h}' for h in hosts}

    def do_GET(self):
        if not self.local_origin(): self.send(403,self.app.session.rejection());return
        if self.path=='/':
            self.send(200,Path(__file__).with_name('dashboard.html').read_bytes(),'text/html; charset=utf-8')
        elif self.path in {'/local/state','/local/state?truth=1'}:
            self.send(200,self.app.state(truth=self.path.endswith('truth=1')))
        elif self.path=='/local/export':
            with self.app.lock: data=self.app.save()
            self.send(200,data,download='local-jammers-run.json')
        elif self.path=='/health': self.send(200,dict(status='ok',kind='local-static-reconstruction'))
        else: self.send(405 if self.path in PATHS else 404,self.app.session.rejection())

    def do_POST(self):
        if not self.local_origin(): self.send(403,self.app.session.rejection());return
        if self.path not in PATHS | {'/local/reset','/local/demo','/local/stop'}:
            self.send(404,self.app.session.rejection());return
        parts=[p.strip().lower() for p in self.headers.get('Content-Type','').split(';')]
        if parts not in [['application/json'],['application/json','charset=utf-8']] or self.headers.get('Content-Encoding','identity').lower()!='identity':
            self.send(415,self.app.session.rejection());return
        if self.headers.get('Transfer-Encoding') is not None or len(self.headers.get_all('Content-Length',[]))!=1:
            self.send(400,self.app.session.rejection());return
        try:
            length=int(self.headers['Content-Length'])
            if length<0: raise ValueError
        except ValueError: self.send(400,self.app.session.rejection());return
        if length>MAX_BODY: self.send(413,self.app.session.rejection());return
        try:
            raw=self.rfile.read(length)
            if len(raw)!=length: raise ValueError('incomplete body')
            payload=decode_json(raw)
        except (ValueError,UnicodeError,RecursionError,TimeoutError):
            self.send(400,self.app.session.rejection());return
        if self.path in PATHS:
            session=self.app.session
            status,response=session.dispatch(self.path,payload)
            self.send(status,response)
            return
        try:
            if self.path=='/local/reset': self.app.reset(payload)
            elif self.path=='/local/demo': self.app.demo()
            else: self.app.cancel.set()
            self.send(200,{'ok':True})
        except (ValueError,TypeError,KeyError) as exc: self.send(400,{'ok':False,'error':str(exc)})
        except RuntimeError as exc: self.send(409,{'ok':False,'error':str(exc)})

    def unsupported(self): self.send(405 if self.path in PATHS else 404,self.app.session.rejection())
    do_PUT=do_DELETE=do_PATCH=do_OPTIONS=do_HEAD=unsupported
