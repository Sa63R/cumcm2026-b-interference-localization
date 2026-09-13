"""Run our Windows automation in the already logged-in guest user's session.
Uses a temporary scheduled task; no Mac keyboard/mouse events or credentials.
"""
import argparse
import base64
from pathlib import Path
import uuid
from utm_guest import ps,call,VM
p=argparse.ArgumentParser();p.add_argument('script',type=Path);a=p.parse_args()
body=a.script.read_text(encoding='utf-8-sig')
destination=r'C:\Users\baiwc\Downloads\Q4V6Lite_20260913'+'\\'+a.script.name
call('file','push',VM,destination,data=body.encode('utf-8-sig'))
loader="& ([scriptblock]::Create([IO.File]::ReadAllText('"+destination+"')))"
encoded=base64.b64encode(loader.encode('utf-16-le')).decode()
name='Codex-Q4V6Lite_20260913-'+uuid.uuid4().hex[:10]
command=f'''$ErrorActionPreference='Stop'
try {{
  $user=(Get-CimInstance Win32_ComputerSystem).UserName
  if (-not $user) {{throw 'No interactive Windows user is logged in'}}
  $action=New-ScheduledTaskAction -Execute 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe' -Argument '-NoProfile -WindowStyle Hidden -EncodedCommand {encoded}' -WorkingDirectory 'C:\\Users\\baiwc\\Downloads\\Q4V6Lite_20260913'
  $principal=New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
  Register-ScheduledTask -TaskName '{name}' -Action $action -Principal $principal | Out-Null
  Start-ScheduledTask -TaskName '{name}'
  @{{task='{name}';user=$user;status='started'}} | ConvertTo-Json | Set-Content -Encoding UTF8 'C:\\Users\\baiwc\\Downloads\\Q4V6Lite_20260913\\desktop_launch.json'
}} catch {{ @{{task='{name}';error=$_.Exception.ToString()}} | ConvertTo-Json | Set-Content -Encoding UTF8 'C:\\Users\\baiwc\\Downloads\\Q4V6Lite_20260913\\desktop_launch.json' }}'''
print(ps(command).decode(errors='replace'))
print('Scheduled',name,'from',a.script)
with (Path(__file__).parent/'results'/'created_tasks.txt').open('a') as f:f.write(name+'\n')
