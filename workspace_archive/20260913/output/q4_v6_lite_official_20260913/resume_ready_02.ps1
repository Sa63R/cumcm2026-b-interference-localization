$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913\additional_10'
if(Test-Path "$root\runs\02"){throw 'Existing run; no resume marker created'}
'Resume existing visibly ready Q4 practice from 02_ready.png; verify canonical case from official completed log.' | Set-Content -Encoding UTF8 "$root\RESUME_READY_02"
