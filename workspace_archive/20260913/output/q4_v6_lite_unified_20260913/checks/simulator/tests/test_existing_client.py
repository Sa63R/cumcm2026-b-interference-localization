"""Integration with this workspace's existing strategy HTTP client, if present."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from jammers_local.core import Scenario,Source
from jammers_local.server import Application,Server

CLIENT_SOURCE=Path(__file__).resolve().parents[2]/'code/src'


@unittest.skipUnless(sys.version_info >= (3,10) and (CLIENT_SOURCE/'simulator_client/client.py').exists(),
                     'workspace client requires Python 3.10+ and the adjacent code/src tree')
class ExistingClientTest(unittest.TestCase):
    def test_existing_http_client_full_session(self):
        sys.path.insert(0,str(CLIENT_SOURCE))
        from simulator_client import SimulatorClient
        from simulator_client.__main__ import smoke_practice
        scene=Scenario(3,(Source(1,0,0,1_000_000_000,'omni',None),),77,profile='fixture')
        app=Application(scene)
        server=Server(app,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                with SimulatorClient('local-test',base_url=f'http://127.0.0.1:{server.server_port}',
                                     log_path=Path(directory)/'client.jsonl') as client:
                    report=smoke_practice(client)
                    self.assertTrue(report['completed'],report.get('error'))
                    self.assertEqual(client.state.virtual_time_s,199)
                    self.assertEqual(client.state.cleared_count,0)
                    self.assertEqual(app.session.engine.time_us,199_000_000)
        finally:
            server.shutdown();server.server_close();thread.join()
