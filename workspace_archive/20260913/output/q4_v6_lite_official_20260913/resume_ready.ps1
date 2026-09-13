$root='C:\Users\baiwc\Downloads\Q4V6Lite_20260913\additional_10'
if(Test-Path "$root\runs\01"){throw 'Existing run; no resume marker created'}
'Start the existing visibly ready Q4 practice KMRK-2X8E-2D6Q-4WSH; no new case.' | Set-Content -Encoding UTF8 "$root\RESUME_READY_01"
