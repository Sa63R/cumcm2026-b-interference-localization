"""Direct documented UTM AppleScript transport; avoids utmctl ScriptingBridge crash."""
import base64,json,subprocess,tempfile,time
from pathlib import Path
VM='755E0E00-7B0E-4845-A9C3-7345C9BC2320'
def q(s):return json.dumps(str(s),ensure_ascii=False)
def script(body):
    src='tell application "UTM"\nset vmRef to first virtual machine whose id is '+q(VM)+'\n'+body+'\nend tell'
    p=subprocess.run(['osascript','-'],input=src.encode(),capture_output=True,check=True)
    return p.stdout

def call(*args,data=None):
    if args[:2]==('file','push'):
        script('set gf to open file vmRef at '+q(args[3])+' for writing\nwrite gf with data '+q(base64.b64encode(data).decode())+' base64 encoding true closing true')
        return b''
    if args[:2]==('file','pull'):
        with tempfile.TemporaryDirectory(prefix='q3q4-pull-') as d:
            p=Path(d)/'data';p.touch()
            for attempt in range(10):
                try:
                    script('set gf to open file vmRef at '+q(args[3])+' for reading\npull gf to POSIX file '+q(p)+' closing true')
                    break
                except subprocess.CalledProcessError:
                    if attempt==9:raise
                    time.sleep(.25)
            return p.read_bytes()
    if args[0]=='exec' and args[2]=='--cmd':
        script('execute vmRef at '+q(args[3])+' with arguments {'+','.join(q(v) for v in args[4:])+'}')
        return b''
    raise ValueError(args[:2])
def ps(body):
    return call('exec',VM,'--cmd',r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(body.encode('utf-16-le')).decode())
