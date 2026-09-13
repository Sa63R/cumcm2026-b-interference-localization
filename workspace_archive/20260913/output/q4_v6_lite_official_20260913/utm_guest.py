"""Host CLI helper for guest-agent file transfer and command execution (no UI)."""
import argparse
import base64
from pathlib import Path
import subprocess

UTM='/Applications/UTM.app/Contents/MacOS/utmctl'
VM='755E0E00-7B0E-4845-A9C3-7345C9BC2320'

def call(*args, data=None):
    p=subprocess.run([UTM,*args],input=data,capture_output=True,check=True)
    return p.stdout

def ps(body):
    return call('exec',VM,'--cmd',r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-EncodedCommand',base64.b64encode(body.encode('utf-16-le')).decode())

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['push','pull','ps']); ap.add_argument('source'); ap.add_argument('destination',nargs='?'); a=ap.parse_args()
    if a.action=='push':
        print(call('file','push',VM,a.destination,data=Path(a.source).read_bytes()).decode(errors='replace'))
    elif a.action=='pull':
        data=call('file','pull',VM,a.source)
        if a.destination:
            Path(a.destination).parent.mkdir(parents=True,exist_ok=True);Path(a.destination).write_bytes(data);print('Saved',len(data),'bytes to',a.destination)
        else:
            print(data.decode('utf-16' if data.startswith(b'\xff\xfe') else 'utf-8',errors='replace'))
    else:
        print(ps(Path(a.source).read_text()).decode(errors='replace'))
