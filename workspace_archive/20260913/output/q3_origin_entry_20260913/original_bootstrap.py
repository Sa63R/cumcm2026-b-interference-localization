from pathlib import Path
import sys,runpy
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'lib'),str(root/'code/experiments/q3_official_practice')]
runpy.run_path(str(root/'code/experiments/q3_official_practice/run_practice.py'),run_name='__main__')
