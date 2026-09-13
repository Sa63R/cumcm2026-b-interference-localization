"""Host-side guest command; does not start/select a UI test mode."""
import argparse
import re
from utm_guest import ps

METHODS=('phased','joint','v2','v3','v3_origin20','optical','scenario','future_cover','scenario_future')
a=argparse.ArgumentParser(description=__doc__)
a.add_argument('--method',choices=METHODS,required=True)
a.add_argument('--robot-id',required=True)
a.add_argument('--case',required=True)
a.add_argument('--practice-confirmed',action='store_true')
v=a.parse_args()
if not v.practice_confirmed: a.error('First visually confirm Q3 practice and its case code.')
if not re.fullmatch(r'[0-9]{1,32}',v.robot_id): a.error('Invalid team ID')
if not re.fullmatch(r'[A-Z0-9-]{1,64}',v.case): a.error('Invalid case label')
root=r'C:\Users\baiwc\Downloads\Q3Practice'
python=r'C:\Users\baiwc\Downloads\Q4Practice\python\python.exe'
command=f'{python} {root}\\bootstrap.py --method {v.method} --robot-id {v.robot_id} --case-label {v.case} --practice-confirmed --output {root}\\practice_results > {root}\\latest_run.log 2>&1'
print(ps(f"& C:\\Windows\\System32\\cmd.exe /c '{command}'").decode(errors='replace'))
print('Dispatched one guest process; verify latest_run.log and result.json for completion.')
